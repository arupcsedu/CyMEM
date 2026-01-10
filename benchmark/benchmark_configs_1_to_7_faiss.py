#!/usr/bin/env python3
"""
benchmark_configs_1_to_7_faiss.py

Benchmark ingestion-style configurations using LlamaIndex + FAISS:
Author: Arup Sarker, djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 10/01/2026

  Set1_Default
    - Sync load (SimpleDirectoryReader)
    - Sync pipeline (IngestionPipeline.run, no workers)
    - Embedding: no concurrency, batch_size=1
    - FAISS add: sequential, batch_size=1

  Set2_ReaderParallel
    - Reader parallel load (SimpleDirectoryReader.load_data(num_workers))
    - Sync pipeline (no workers)
    - Embedding: no concurrency, batch_size=1
    - FAISS add: sequential, batch_size=1

  Set3_PipelineParallelSync
    - Sync load
    - Sync pipeline with multiprocessing (run(num_workers=pipeline_workers))
    - Embedding: no concurrency, batch_size=1
    - FAISS add: sequential, batch_size=1

  Set4_AsyncOnly
    - Sync load
    - Async pipeline (arun(num_workers=async_workers))
    - Embedding: async concurrency (num_workers=async_workers), batch_size=1
    - FAISS add: sequential, batch_size=1

  Set5_AsyncPlusBatching
    - Sync load
    - Async pipeline
    - Embedding: async concurrency, batched (set5_embed_batch)
    - FAISS add: sequential batched add (set5_add_batch)

  Set6_BSP_ShardedFAISS
    - Superstep A: load + async transform
    - Superstep B: async batched embedding (bsp_embed_batch)
    - Superstep C: sharded FAISS build in parallel (BSP-style)

  Set7_BAPP_BulkAsyncParallel
    - Sync load
    - Async pipeline
    - Embedding + FAISS add overlapped in a bulk async pipeline:
        • multiple async embedding workers
        • single FAISS writer
        • batches flow through queues (no global barrier between embed + add)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import random
import string
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import faiss

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode, TransformComponent

# Delimiter used for our synthetic node splitter
DELIM = "\n\n<<<NODE_SPLIT>>>\n\n"


# ---------------------------------------------------------------------------
# Synthetic corpus generation
# ---------------------------------------------------------------------------

def _rand_text(chars: int, seed: int) -> str:
    rnd = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + "     "
    return "".join(rnd.choice(alphabet) for _ in range(chars)).strip()


def write_synthetic_corpus(
    data_dir: str,
    nodes: int,
    node_chars: int,
    num_files: int,
    seed: int,
) -> None:
    """
    Create `num_files` text files in `data_dir` such that the total number
    of logical chunks (separated by DELIM) equals exactly `nodes`.
    """
    os.makedirs(data_dir, exist_ok=True)
    base = nodes // num_files
    rem = nodes % num_files
    per_file = [base + (1 if i < rem else 0) for i in range(num_files)]

    idx = 0
    for fi, k in enumerate(per_file):
        parts = []
        for _ in range(k):
            parts.append(_rand_text(node_chars, seed + idx))
            idx += 1
        content = DELIM.join(parts)
        with open(os.path.join(data_dir, f"doc_{fi:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# Transform: delimiter splitter (must be TransformComponent for IngestionPipeline)
# ---------------------------------------------------------------------------

class DelimiterNodeSplitter(TransformComponent):
    """
    Splits incoming Documents/Nodes by a delimiter and emits TextNodes.

    This must inherit from TransformComponent so that IngestionPipeline
    accepts it (pydantic validation).
    """
    delimiter: str = DELIM

    def __call__(self, nodes, **kwargs):
        out: List[TextNode] = []
        for doc_i, obj in enumerate(nodes):
            text = getattr(obj, "text", None)
            if text is None and hasattr(obj, "get_text"):
                text = obj.get_text()
            if text is None:
                text = str(obj)

            chunks = [c for c in text.split(self.delimiter) if c.strip()]
            base_id = getattr(obj, "doc_id", None) or getattr(obj, "id_", None) or f"item{doc_i}"
            base_meta = {}
            if hasattr(obj, "metadata") and isinstance(obj.metadata, dict):
                base_meta = dict(obj.metadata)

            for chunk_i, chunk in enumerate(chunks):
                node_id = f"{base_id}-chunk{chunk_i}"
                meta = dict(base_meta)
                meta.update({"doc_index": doc_i, "chunk_index": chunk_i})
                out.append(TextNode(id_=node_id, text=chunk, metadata=meta))
        return out


# ---------------------------------------------------------------------------
# Fake embedder (API-like latency model)
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """
    Simulates a remote embedding API:

      latency(batch) = request_overhead_ms + per_item_ms * len(batch)

    and returns deterministic float32 vectors derived from SHA256(text).
    """

    def __init__(self, dim: int, request_overhead_ms: float, per_item_ms: float):
        self.dim = dim
        self.request_overhead_ms = request_overhead_ms
        self.per_item_ms = per_item_ms

    async def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        sleep_s = (self.request_overhead_ms + self.per_item_ms * len(texts)) / 1000.0
        await asyncio.sleep(sleep_s)

        arr = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            rep = int(np.ceil(self.dim / len(v)))
            vv = np.tile(v, rep)[: self.dim]
            arr[i] = (vv / 127.5) - 1.0
        return arr


async def embed_all_async(embedder: FakeEmbedder, texts: Sequence[str],
                          batch_size: int, num_workers: int) -> np.ndarray:
    sem = asyncio.Semaphore(num_workers)

    async def _run(batch: Sequence[str]) -> np.ndarray:
        async with sem:
            return await embedder.embed_batch(batch)

    tasks = []
    for s in range(0, len(texts), batch_size):
        tasks.append(asyncio.create_task(_run(texts[s:s + batch_size])))

    chunks = await asyncio.gather(*tasks)
    return np.vstack(chunks) if chunks else np.empty((0, embedder.dim), dtype=np.float32)


def chunk_ranges(n: int, batch_size: int) -> List[Tuple[int, int]]:
    return [(i, min(n, i + batch_size)) for i in range(0, n, batch_size)]


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

class FaissFlatStore:
    """Single-process FAISS flat index (IndexFlatL2 or IndexFlatIP)."""

    def __init__(self, dim: int, metric: str = "ip"):
        self.dim = dim
        if metric == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add_batch(self, x: np.ndarray) -> None:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        self.index.add(np.ascontiguousarray(x))

    @property
    def ntotal(self) -> int:
        return self.index.ntotal


class FaissShardedStore:
    """
    BSP-friendly FAISS store: builds multiple IndexFlat shards, then combines
    them into an IndexShards instance.
    """

    def __init__(self, dim: int, shards: int, metric: str = "ip"):
        self.dim = dim
        self.shards = shards
        self.metric = metric
        self.shard_indexes = [
            faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)
            for _ in range(shards)
        ]
        self.combined: Optional[faiss.Index] = None

    def _add_to_shard(self, shard_id: int, x: np.ndarray) -> None:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        self.shard_indexes[shard_id].add(np.ascontiguousarray(x))

    def finalize(self) -> faiss.Index:
        sh = faiss.IndexShards(self.dim, True, False)
        for idx in self.shard_indexes:
            sh.add_shard(idx)
        self.combined = sh
        return sh

    @property
    def ntotal(self) -> int:
        if self.combined is not None:
            return self.combined.ntotal
        return sum(i.ntotal for i in self.shard_indexes)


def faiss_add_sequential(index: FaissFlatStore, x: np.ndarray, batch_size: int) -> float:
    """Sequential FAISS add in batches (used by Set1–5)."""
    t0 = time.perf_counter()
    n = x.shape[0]
    for s, e in chunk_ranges(n, batch_size):
        index.add_batch(x[s:e])
    return time.perf_counter() - t0


async def faiss_add_bsp_sharded(
    store: FaissShardedStore,
    x: np.ndarray,
    shard_workers: int,
) -> float:
    """
    BSP-style sharded FAISS build (Set6):
      - partition x across shards
      - build each shard in parallel (threads)
      - barrier
      - combine shards via IndexShards
    """
    t0 = time.perf_counter()

    n = x.shape[0]
    shards = store.shards
    base = n // shards
    rem = n % shards
    parts: List[Tuple[int, int, int]] = []
    start = 0
    for i in range(shards):
        k = base + (1 if i < rem else 0)
        parts.append((i, start, start + k))
        start += k

    sem = asyncio.Semaphore(shard_workers)

    async def _build_shard(shard_id: int, s: int, e: int):
        async with sem:
            await asyncio.to_thread(store._add_to_shard, shard_id, x[s:e])

    tasks = [asyncio.create_task(_build_shard(i, s, e)) for (i, s, e) in parts]
    await asyncio.gather(*tasks)  # barrier
    store.finalize()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# BAPP (Bulk Asynchronous Parallel Processing) for FAISS
# ---------------------------------------------------------------------------

async def run_bapp_faiss(
    nodes: List[TextNode],
    embedder: FakeEmbedder,
    index: FaissFlatStore,
    embed_batch_size: int,
    embed_workers: int,
) -> Tuple[float, float, float]:
    """
    Bulk Asynchronous Parallel Processing for FAISS (Set7):

      - Split nodes into batches.
      - Producer enqueues (start, end) index ranges.
      - Multiple async embedding workers consume ranges, call embed_batch(),
        and enqueue embedding batches into an upsert queue.
      - Single consumer takes embedding batches and calls index.add_batch(),
        ensuring FAISS is written by one logical writer.
      - Embedding and FAISS add overlap in time (no global barrier).

    Returns:
        (embed_wall_time, upsert_wall_time, stage_total_wall_time)
    """
    texts = [n.text for n in nodes]
    n = len(texts)

    embed_queue: asyncio.Queue[Optional[Tuple[int, int]]] = asyncio.Queue()
    upsert_queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue()

    embed_start: Optional[float] = None
    embed_end: Optional[float] = None
    upsert_start: Optional[float] = None
    upsert_end: Optional[float] = None
    time_lock = asyncio.Lock()

    async def producer():
        # Put index ranges for embedding
        for s, e in chunk_ranges(n, embed_batch_size):
            await embed_queue.put((s, e))
        # One sentinel per embedding worker
        for _ in range(embed_workers):
            await embed_queue.put(None)

    async def embed_worker():
        nonlocal embed_start, embed_end
        while True:
            item = await embed_queue.get()
            if item is None:
                # signal consumer that one worker is done
                await upsert_queue.put(None)
                break
            s, e = item
            async with time_lock:
                if embed_start is None:
                    embed_start = time.perf_counter()
            X_batch = await embedder.embed_batch(texts[s:e])
            async with time_lock:
                embed_end = time.perf_counter()
            await upsert_queue.put(X_batch)

    async def consumer_upsert():
        nonlocal upsert_start, upsert_end
        finished_embed_workers = 0
        while True:
            item = await upsert_queue.get()
            if item is None:
                finished_embed_workers += 1
                if finished_embed_workers == embed_workers:
                    break
                continue
            X_batch = item
            async with time_lock:
                if upsert_start is None:
                    upsert_start = time.perf_counter()
            index.add_batch(X_batch)
            async with time_lock:
                upsert_end = time.perf_counter()

    stage_start = time.perf_counter()

    prod_task = asyncio.create_task(producer())
    embed_tasks = [asyncio.create_task(embed_worker()) for _ in range(embed_workers)]
    cons_task = asyncio.create_task(consumer_upsert())

    await asyncio.gather(prod_task, *embed_tasks, cons_task)
    stage_end = time.perf_counter()

    embed_s = (embed_end - embed_start) if (embed_start is not None and embed_end is not None) else 0.0
    upsert_s = (upsert_end - upsert_start) if (upsert_start is not None and upsert_end is not None) else 0.0
    total_s = stage_end - stage_start
    return embed_s, upsert_s, total_s


# ---------------------------------------------------------------------------
# Benchmark rows + pretty table
# ---------------------------------------------------------------------------

@dataclass
class ResultRow:
    config: str
    nodes: int
    load_s: float
    transform_s: float
    embed_s: float
    upsert_s: float
    total_s: float


def pct_faster(base: float, new: float) -> float:
    return (base - new) / base * 100.0 if base > 0 else 0.0


def print_table(rows: List[ResultRow]) -> None:
    headers = ["Config", "Nodes", "Load(s)", "Transform(s)",
               "Embed(s)", "FAISS Add(s)", "Total(s)", "Δ vs Set1"]
    base = rows[0].total_s if rows else 0.0

    data: List[List[str]] = []
    for r in rows:
        delta = "baseline" if r.config == "Set1_Default" else f"{pct_faster(base, r.total_s):.1f}% faster"
        data.append([
            r.config,
            str(r.nodes),
            f"{r.load_s:.3f}",
            f"{r.transform_s:.3f}",
            f"{r.embed_s:.3f}",
            f"{r.upsert_s:.3f}",
            f"{r.total_s:.3f}",
            delta,
        ])

    cols = list(zip(headers, *data))
    widths = [max(len(x) for x in col) for col in cols]

    def fmt(items: Sequence[str]) -> str:
        return " | ".join(items[i].ljust(widths[i]) for i in range(len(items)))

    sep = "-+-".join("-" * w for w in widths)
    print()
    print(fmt(headers))
    print(sep)
    for row in data:
        print(fmt(row))
    print()


# ---------------------------------------------------------------------------
# Load / Transform helpers
# ---------------------------------------------------------------------------

def load_docs_sync(data_dir: str):
    t0 = time.perf_counter()
    docs = SimpleDirectoryReader(data_dir).load_data()
    return docs, time.perf_counter() - t0


def load_docs_parallel(data_dir: str, num_workers: int):
    t0 = time.perf_counter()
    docs = SimpleDirectoryReader(data_dir).load_data(num_workers=num_workers)
    return docs, time.perf_counter() - t0


def transform_sync(docs, num_workers: Optional[int]):
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])
    t0 = time.perf_counter()
    nodes = pipeline.run(documents=docs, num_workers=num_workers)
    return nodes, time.perf_counter() - t0


async def transform_async(docs, num_workers: int):
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])
    t0 = time.perf_counter()
    nodes = await pipeline.arun(documents=docs, num_workers=num_workers)
    return nodes, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark LlamaIndex + FAISS configs 1–7.")
    p.add_argument("--nodes", type=int, default=20000,
                   help="Total number of logical nodes (chunks).")
    p.add_argument("--node-chars", type=int, default=800,
                   help="Characters per node chunk.")
    p.add_argument("--files", type=int, default=200,
                   help="Number of files for SimpleDirectoryReader.")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--reader-workers", type=int, default=8,
                   help="Set2 reader num_workers.")
    p.add_argument("--pipeline-workers", type=int, default=8,
                   help="Set3 pipeline num_workers.")
    p.add_argument("--async-workers", type=int, default=32,
                   help="Async pipeline + embedding concurrency.")

    p.add_argument("--set5-embed-batch", type=int, default=64,
                   help="Set5 embedding batch size.")
    p.add_argument("--set5-add-batch", type=int, default=4096,
                   help="Set5 FAISS add batch size.")

    # BSP (Set6) knobs
    p.add_argument("--bsp-embed-batch", type=int, default=128,
                   help="Set6 embedding batch size.")
    p.add_argument("--bsp-shards", type=int, default=8,
                   help="Set6 number of FAISS shards.")
    p.add_argument("--bsp-shard-workers", type=int, default=8,
                   help="Set6 shard build workers (async semaphore).")

    # BAPP (Set7) knobs
    p.add_argument("--bapp-embed-batch", type=int, default=128,
                   help="Set7 embedding batch size.")
    p.add_argument("--bapp-embed-workers", type=int, default=8,
                   help="Set7 number of async embedding workers.")

    p.add_argument("--dim", type=int, default=768,
                   help="Embedding / FAISS dimension.")
    p.add_argument("--metric", choices=["ip", "l2"], default="ip",
                   help="FAISS metric: inner-product or L2.")

    p.add_argument("--request-overhead-ms", type=float, default=60.0,
                   help="Fake embedding fixed overhead per request (ms).")
    p.add_argument("--per-item-ms", type=float, default=1.2,
                   help="Fake embedding per-item latency (ms).")

    p.add_argument("--faiss-omp-threads", type=int, default=8,
                   help="FAISS OpenMP thread count.")
    return p.parse_args()


async def main_async():
    args = parse_args()

    # Configure FAISS threads
    try:
        faiss.omp_set_num_threads(int(args.faiss_omp_threads))
    except Exception:
        pass

    embedder = FakeEmbedder(args.dim, args.request_overhead_ms, args.per_item_ms)
    rows: List[ResultRow] = []

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        write_synthetic_corpus(data_dir, args.nodes, args.node_chars, args.files, args.seed)

        # ---------------- Set 1: Default ----------------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = transform_sync(docs, num_workers=None)

        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(embedder, texts, batch_size=1, num_workers=1)
        embed_s = time.perf_counter() - t0

        index = FaissFlatStore(args.dim, metric=args.metric)
        add_s = faiss_add_sequential(index, X, batch_size=1)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set1_Default", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # ---------------- Set 2: Reader Parallel ----------------
        t_all = time.perf_counter()
        docs, load_s = load_docs_parallel(data_dir, args.reader_workers)
        nodes, transform_s = transform_sync(docs, num_workers=None)

        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(embedder, texts, batch_size=1, num_workers=1)
        embed_s = time.perf_counter() - t0

        index = FaissFlatStore(args.dim, metric=args.metric)
        add_s = faiss_add_sequential(index, X, batch_size=1)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set2_ReaderParallel", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # ---------------- Set 3: Pipeline Parallel (Sync) ----------------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = transform_sync(docs, num_workers=args.pipeline_workers)

        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(embedder, texts, batch_size=1, num_workers=1)
        embed_s = time.perf_counter() - t0

        index = FaissFlatStore(args.dim, metric=args.metric)
        add_s = faiss_add_sequential(index, X, batch_size=1)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set3_PipelineParallelSync", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # ---------------- Set 4: Async Only ----------------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)

        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(embedder, texts, batch_size=1, num_workers=args.async_workers)
        embed_s = time.perf_counter() - t0

        index = FaissFlatStore(args.dim, metric=args.metric)
        add_s = faiss_add_sequential(index, X, batch_size=1)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set4_AsyncOnly", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # ---------------- Set 5: Async + Batching ----------------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)

        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(
            embedder, texts,
            batch_size=args.set5_embed_batch,
            num_workers=args.async_workers,
        )
        embed_s = time.perf_counter() - t0

        index = FaissFlatStore(args.dim, metric=args.metric)
        add_s = faiss_add_sequential(index, X, batch_size=args.set5_add_batch)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set5_AsyncPlusBatching", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # ---------------- Set 6: BSP (Sharded FAISS) ----------------
        # Superstep A: load + async transform
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)

        # Superstep B: batched embeddings (async)
        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(
            embedder, texts,
            batch_size=args.bsp_embed_batch,
            num_workers=args.async_workers,
        )
        embed_s = time.perf_counter() - t0

        # Superstep C: sharded FAISS build in parallel
        sharded = FaissShardedStore(args.dim, shards=args.bsp_shards, metric=args.metric)
        add_s = await faiss_add_bsp_sharded(sharded, X, shard_workers=args.bsp_shard_workers)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set6_BSP_ShardedFAISS", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # ---------------- Set 7: BAPP (Bulk Async Parallel Processing) ----------------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)

        index = FaissFlatStore(args.dim, metric=args.metric)
        embed_s, add_s, _stage_total = await run_bapp_faiss(
            nodes=nodes,
            embedder=embedder,
            index=index,
            embed_batch_size=args.bapp_embed_batch,
            embed_workers=args.bapp_embed_workers,
        )
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set7_BAPP_BulkAsyncParallel", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # ---------------- Print results ----------------
        print_table(rows)

        r4 = next(r for r in rows if r.config == "Set4_AsyncOnly")
        r5 = next(r for r in rows if r.config == "Set5_AsyncPlusBatching")
        r6 = next(r for r in rows if r.config == "Set6_BSP_ShardedFAISS")
        r7 = next(r for r in rows if r.config == "Set7_BAPP_BulkAsyncParallel")

        print(f"Set5 vs Set4: {pct_faster(r4.total_s, r5.total_s):.1f}% faster")
        print(f"Set6 vs Set5: {pct_faster(r5.total_s, r6.total_s):.1f}% faster")
        print(f"Set7 vs Set5: {pct_faster(r5.total_s, r7.total_s):.1f}% faster")
        print(f"Set7 vs Set6: {pct_faster(r6.total_s, r7.total_s):.1f}% faster")
        print(f"Set7 vs Set4: {pct_faster(r4.total_s, r7.total_s):.1f}% faster\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

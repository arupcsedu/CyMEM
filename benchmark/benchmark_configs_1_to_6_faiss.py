#!/usr/bin/env python3

#Author: Arup Sarker, djy8hg@virginia.edu, arupcsedu@gmail.com
#Date: 10/01/2026
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

DELIM = "\n\n<<<NODE_SPLIT>>>\n\n"


# -------------------------
# Synthetic corpus generation
# -------------------------

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


# -------------------------
# Transform: delimiter splitter (TransformComponent required)
# -------------------------

class DelimiterNodeSplitter(TransformComponent):
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


# -------------------------
# Fake embedder (API-like latency)
# -------------------------

class FakeEmbedder:
    def __init__(self, dim: int, request_overhead_ms: float, per_item_ms: float):
        self.dim = dim
        self.request_overhead_ms = request_overhead_ms
        self.per_item_ms = per_item_ms

    async def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        sleep_s = (self.request_overhead_ms + self.per_item_ms * len(texts)) / 1000.0
        await asyncio.sleep(sleep_s)

        # Deterministic vectors via SHA256 -> float32
        arr = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Fill dim using repeating bytes mapped to [-1, 1]
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            # Repeat/trim to dim
            rep = int(np.ceil(self.dim / len(v)))
            vv = np.tile(v, rep)[: self.dim]
            arr[i] = (vv / 127.5) - 1.0
        return arr


async def embed_all_async(embedder: FakeEmbedder, texts: Sequence[str], batch_size: int, num_workers: int) -> np.ndarray:
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


# -------------------------
# FAISS store(s)
# -------------------------

class FaissFlatStore:
    """Single FAISS index (not safe for concurrent add from multiple tasks)."""
    def __init__(self, dim: int, metric: str = "ip"):
        self.dim = dim
        if metric == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add_batch(self, x: np.ndarray) -> None:
        # x must be float32 contiguous
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        self.index.add(np.ascontiguousarray(x))

    @property
    def ntotal(self) -> int:
        return self.index.ntotal


class FaissShardedStore:
    """
    BSP-friendly: build shards in parallel, then combine using IndexShards.
    Each shard is its own IndexFlat*, safe because shards are independent.
    """
    def __init__(self, dim: int, shards: int, metric: str = "ip"):
        self.dim = dim
        self.shards = shards
        self.metric = metric
        self.shard_indexes = [
            faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)
            for _ in range(shards)
        ]
        self.combined = None  # faiss.IndexShards

    def _add_to_shard(self, shard_id: int, x: np.ndarray) -> None:
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        self.shard_indexes[shard_id].add(np.ascontiguousarray(x))

    def finalize(self) -> faiss.Index:
        sh = faiss.IndexShards(self.dim, True, False)  # (d, threaded, successive_ids)
        for idx in self.shard_indexes:
            sh.add_shard(idx)
        self.combined = sh
        return sh

    @property
    def ntotal(self) -> int:
        if self.combined is not None:
            return self.combined.ntotal
        return sum(i.ntotal for i in self.shard_indexes)


# -------------------------
# Benchmark rows + printing
# -------------------------

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
    headers = ["Config", "Nodes", "Load(s)", "Transform(s)", "Embed(s)", "FAISS Add(s)", "Total(s)", "Δ vs Set1"]
    base = rows[0].total_s if rows else 0.0

    data = []
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

    print("\n" + fmt(headers))
    print(sep)
    for row in data:
        print(fmt(row))
    print()


# -------------------------
# Load/Transform helpers
# -------------------------

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


# -------------------------
# FAISS add implementations
# -------------------------

def faiss_add_sequential(index: FaissFlatStore, x: np.ndarray, batch_size: int) -> float:
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
    BSP-style parallel upsert:
      - Partition x across shards
      - Build each shard in parallel (threads)
      - Barrier
      - Finalize combined IndexShards
    """
    t0 = time.perf_counter()

    n = x.shape[0]
    shards = store.shards
    # Even partition of rows into shards
    parts = []
    base = n // shards
    rem = n % shards
    start = 0
    for i in range(shards):
        k = base + (1 if i < rem else 0)
        parts.append((i, start, start + k))
        start += k

    sem = asyncio.Semaphore(shard_workers)

    async def _build_shard(shard_id: int, s: int, e: int):
        async with sem:
            # run CPU add in thread so event loop stays responsive
            await asyncio.to_thread(store._add_to_shard, shard_id, x[s:e])

    tasks = [asyncio.create_task(_build_shard(i, s, e)) for (i, s, e) in parts]
    await asyncio.gather(*tasks)  # barrier
    store.finalize()              # barrier boundary for next step
    return time.perf_counter() - t0


# -------------------------
# Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=32)
    p.add_argument("--node-chars", type=int, default=800)
    p.add_argument("--files", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--reader-workers", type=int, default=8)
    p.add_argument("--pipeline-workers", type=int, default=8)
    p.add_argument("--async-workers", type=int, default=32)

    p.add_argument("--set5-embed-batch", type=int, default=64)
    p.add_argument("--set5-add-batch", type=int, default=4096)

    # BSP (FAISS sharded) knobs
    p.add_argument("--bsp-embed-batch", type=int, default=128)
    p.add_argument("--bsp-shards", type=int, default=8)
    p.add_argument("--bsp-shard-workers", type=int, default=8)

    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--metric", choices=["ip", "l2"], default="ip")

    # Fake embedder model
    p.add_argument("--request-overhead-ms", type=float, default=60.0)
    p.add_argument("--per-item-ms", type=float, default=1.2)

    # FAISS threading (OpenMP)
    p.add_argument("--faiss-omp-threads", type=int, default=8)
    return p.parse_args()


async def main_async():
    args = parse_args()

    # FAISS CPU threading (helps search, sometimes helps add depending on build)
    try:
        faiss.omp_set_num_threads(int(args.faiss_omp_threads))
    except Exception:
        pass

    embedder = FakeEmbedder(args.dim, args.request_overhead_ms, args.per_item_ms)

    rows: List[ResultRow] = []

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        write_synthetic_corpus(data_dir, args.nodes, args.node_chars, args.files, args.seed)

        # -------- Set 1: Default --------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = transform_sync(docs, num_workers=None)

        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(embedder, texts, batch_size=1, num_workers=1)  # no concurrency
        embed_s = time.perf_counter() - t0

        index = FaissFlatStore(args.dim, metric=args.metric)
        add_s = faiss_add_sequential(index, X, batch_size=1)  # no batching
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set1_Default", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # -------- Set 2: Reader Parallel --------
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

        # -------- Set 3: Pipeline Parallel (Sync) --------
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

        # -------- Set 4: Async only (concurrency, no batching) --------
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

        # -------- Set 5: Async + batching (embed batch + FAISS bulk add) --------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)

        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(embedder, texts, batch_size=args.set5_embed_batch, num_workers=args.async_workers)
        embed_s = time.perf_counter() - t0

        index = FaissFlatStore(args.dim, metric=args.metric)
        add_s = faiss_add_sequential(index, X, batch_size=args.set5_add_batch)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set5_AsyncPlusBatching", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        # -------- Set 6: BSP (sharded FAISS build in parallel) --------
        # Superstep A: load+transform (barrier)
        # Superstep B: embedding in batches (barrier)
        # Superstep C: build shards in parallel, then combine IndexShards (barrier)
        t_all = time.perf_counter()

        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)

        texts = [n.text for n in nodes]
        t0 = time.perf_counter()
        X = await embed_all_async(embedder, texts, batch_size=args.bsp_embed_batch, num_workers=args.async_workers)
        embed_s = time.perf_counter() - t0

        sharded = FaissShardedStore(args.dim, shards=args.bsp_shards, metric=args.metric)
        add_s = await faiss_add_bsp_sharded(sharded, X, shard_workers=args.bsp_shard_workers)

        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set6_BSP_SharededFAISS", len(nodes), load_s, transform_s, embed_s, add_s, total_s))

        print_table(rows)

        # Quick deltas
        r4 = next(r for r in rows if r.config == "Set4_AsyncOnly")
        r5 = next(r for r in rows if r.config == "Set5_AsyncPlusBatching")
        r6 = next(r for r in rows if r.config == "Set6_BSP_SharededFAISS")
        print(f"Set5 vs Set4: {pct_faster(r4.total_s, r5.total_s):.1f}% faster")
        print(f"Set6 vs Set5: {pct_faster(r5.total_s, r6.total_s):.1f}% faster")
        print(f"Set6 vs Set4: {pct_faster(r4.total_s, r6.total_s):.1f}% faster\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

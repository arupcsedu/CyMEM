#!/usr/bin/env python3
"""
Benchmark LlamaIndex ingestion configs (Set 1..5) with configurable node count.
Author: Arup Sarker, djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 06/01/2026

Implements:
  Set 1: Default
    - Sync load (SimpleDirectoryReader.load_data())
    - Sync pipeline (IngestionPipeline.run()) sequential transforms
    - Embed + upsert sequential, no batching

  Set 2: Reader Parallel
    - Parallel reader load (SimpleDirectoryReader.load_data(num_workers=8)) :contentReference[oaicite:1]{index=1}
    - Sync pipeline sequential transforms
    - Embed + upsert sequential, no batching

  Set 3: Pipeline Parallel (Sync)
    - Sync reader load
    - Sync pipeline with multiprocessing (IngestionPipeline.run(num_workers=K)) :contentReference[oaicite:2]{index=2}
    - Embed + upsert sequential, no batching

  Set 4: Async Pipeline (concurrency only)
    - Sync reader load
    - Async pipeline transforms (IngestionPipeline.arun)
    - Embed: async with concurrency (num_workers), BUT embed_batch_size=1
    - Upsert: upsert_batch_size=1

  Set 5: Async + Batching
    - Sync reader load
    - Async pipeline transforms
    - Embed: async + batching (embed_batch_size configurable)
    - Upsert: batched inserts (upsert_batch_size configurable)

Notes:
- We use a delimiter-based splitter transform so node count is exactly controllable.
- We simulate embedding latency to show batching benefits reliably.

Dependencies:
  pip install llama-index chromadb
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
from typing import Any, Dict, List, Sequence, Tuple

import chromadb

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode, TransformComponent


# -------------------------
# Synthetic data generation
# -------------------------

DELIM = "\n\n<<<NODE_SPLIT>>>\n\n"

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
    Writes num_files text files. The total number of node-chunks across all files is exactly `nodes`.
    Each node-chunk is separated by DELIM. Our splitter transform will create exactly one TextNode per chunk.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Distribute nodes across files as evenly as possible
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
# LlamaIndex Transform: Delimiter splitter
# -------------------------

class DelimiterNodeSplitter(TransformComponent):
    """
    Custom splitter compatible with IngestionPipeline.
    Must inherit TransformComponent (pydantic model) per LlamaIndex docs. :contentReference[oaicite:1]{index=1}
    Splits each incoming item (Document or Node-like) by a delimiter and returns TextNodes.
    """
    delimiter: str = DELIM  # pydantic field

    def __call__(self, nodes, **kwargs):
        out: List[TextNode] = []

        for doc_i, obj in enumerate(nodes):
            # LlamaIndex may pass Documents first, then nodes through subsequent transforms.
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
# Fake embedding (API-like)
# -------------------------

class FakeEmbedder:
    """
    Simulates remote embedding API:
      latency(batch) = request_overhead_ms + per_item_ms * len(batch)
    Returns deterministic vectors of length `dim` derived from SHA256(text).
    """
    def __init__(self, dim: int, request_overhead_ms: float, per_item_ms: float):
        self.dim = dim
        self.request_overhead_ms = request_overhead_ms
        self.per_item_ms = per_item_ms

    async def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        sleep_s = (self.request_overhead_ms + self.per_item_ms * len(texts)) / 1000.0
        await asyncio.sleep(sleep_s)

        out: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = []
            for j in range(self.dim):
                b = h[j % len(h)]
                vec.append((b / 127.5) - 1.0)  # [-1, 1]
            out.append(vec)
        return out


# -------------------------
# Vector DB: Chroma helpers
# -------------------------

def init_chroma(persist_dir: str | None, collection_name: str) -> Any:
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.EphemeralClient()

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


async def chroma_upsert_batches(
    collection: Any,
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Sequence[Dict[str, Any]],
    documents: Sequence[str],
    upsert_batch_size: int,
) -> None:
    n = len(ids)
    for start in range(0, n, upsert_batch_size):
        end = min(n, start + upsert_batch_size)
        collection.add(
            ids=list(ids[start:end]),
            embeddings=[list(x) for x in embeddings[start:end]],
            metadatas=list(metadatas[start:end]),
            documents=list(documents[start:end]),
        )
        await asyncio.sleep(0)


# -------------------------
# Async embedding runner
# -------------------------

async def embed_all_async(
    embedder: FakeEmbedder,
    texts: Sequence[str],
    embed_batch_size: int,
    num_workers: int,
) -> List[List[float]]:
    sem = asyncio.Semaphore(num_workers)

    async def _run_batch(batch: Sequence[str]) -> List[List[float]]:
        async with sem:
            return await embedder.embed_batch(batch)

    tasks = []
    for start in range(0, len(texts), embed_batch_size):
        batch = texts[start:start + embed_batch_size]
        tasks.append(asyncio.create_task(_run_batch(batch)))

    # preserve order
    batches = await asyncio.gather(*tasks)
    out: List[List[float]] = []
    for b in batches:
        out.extend(b)
    return out


# -------------------------
# Benchmark data structures
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


def pct_faster(baseline: float, new: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - new) / baseline * 100.0


# -------------------------
# Config implementations
# -------------------------

def load_docs_sync(data_dir: str) -> Tuple[List[Any], float]:
    t0 = time.perf_counter()
    docs = SimpleDirectoryReader(data_dir).load_data()
    t1 = time.perf_counter()
    return docs, (t1 - t0)

def load_docs_parallel(data_dir: str, num_workers: int) -> Tuple[List[Any], float]:
    # SimpleDirectoryReader parallel load example uses load_data(num_workers=...) :contentReference[oaicite:3]{index=3}
    t0 = time.perf_counter()
    reader = SimpleDirectoryReader(data_dir)
    docs = reader.load_data(num_workers=num_workers)
    t1 = time.perf_counter()
    return docs, (t1 - t0)

def transform_sync(docs: List[Any], num_workers: int | None) -> Tuple[List[TextNode], float]:
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])

    t0 = time.perf_counter()
    # IngestionPipeline.run supports parallel processing with num_workers (multiprocessing.Pool) :contentReference[oaicite:4]{index=4}
    nodes = pipeline.run(documents=docs, num_workers=num_workers)
    t1 = time.perf_counter()
    return nodes, (t1 - t0)

async def transform_async(docs: List[Any], num_workers: int) -> Tuple[List[TextNode], float]:
    splitter = DelimiterNodeSplitter()
    pipeline = IngestionPipeline(transformations=[splitter])

    t0 = time.perf_counter()
    # Async pipeline exists; num_workers controls outgoing concurrency as semaphore in async examples :contentReference[oaicite:5]{index=5}
    nodes = await pipeline.arun(documents=docs, num_workers=num_workers)
    t1 = time.perf_counter()
    return nodes, (t1 - t0)

async def embed_and_upsert(
    nodes: List[TextNode],
    embedder: FakeEmbedder,
    collection: Any,
    embed_batch_size: int,
    upsert_batch_size: int,
    embed_num_workers: int,
) -> Tuple[float, float]:
    texts = [n.text for n in nodes]
    ids = [n.id_ for n in nodes]
    metas = [dict(n.metadata or {}) for n in nodes]

    te0 = time.perf_counter()
    embs = await embed_all_async(embedder, texts, embed_batch_size=embed_batch_size, num_workers=embed_num_workers)
    te1 = time.perf_counter()

    tu0 = time.perf_counter()
    await chroma_upsert_batches(
        collection=collection,
        ids=ids,
        embeddings=embs,
        metadatas=metas,
        documents=texts,
        upsert_batch_size=upsert_batch_size,
    )
    tu1 = time.perf_counter()

    return (te1 - te0), (tu1 - tu0)


# -------------------------
# Pretty table
# -------------------------

def print_table(rows: List[ResultRow]) -> None:
    headers = ["Config", "Nodes", "Load(s)", "Transform(s)", "Embed(s)", "Upsert(s)", "Total(s)", "Δ vs Set1"]
    set1_total = rows[0].total_s if rows else 0.0

    # Build formatted strings
    data = []
    for r in rows:
        delta = f"{pct_faster(set1_total, r.total_s):.1f}% faster" if r.config != "Set1_Default" else "baseline"
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

    # Column widths
    cols = list(zip(headers, *data))
    widths = [max(len(x) for x in col) for col in cols]

    def fmt_row(items: Sequence[str]) -> str:
        return " | ".join(s.ljust(w) for s, w in zip(items, widths))

    sep = "-+-".join("-" * w for w in widths)

    print("\n" + fmt_row(headers))
    print(sep)
    for row in data:
        print(fmt_row(row))
    print()


# -------------------------
# Main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LlamaIndex ingestion configs Set1..Set5.")
    p.add_argument("--nodes", type=int, default=32, help="Exact number of nodes to create.")
    p.add_argument("--node-chars", type=int, default=900, help="Characters per node (controls text size).")
    p.add_argument("--files", type=int, default=100, help="Number of files for SimpleDirectoryReader to load.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")

    # Reader/pipeline parallelism
    p.add_argument("--reader-workers", type=int, default=8, help="Set2 reader parallel workers.")
    p.add_argument("--pipeline-workers", type=int, default=8, help="Set3 pipeline multiprocessing workers.")

    # Async concurrency
    p.add_argument("--async-workers", type=int, default=32, help="Set4/5 async concurrency for transforms and embeddings.")

    # Set5 batching
    p.add_argument("--set5-embed-batch", type=int, default=64, help="Set5 embedding batch size.")
    p.add_argument("--set5-upsert-batch", type=int, default=256, help="Set5 upsert batch size.")

    # Fake embedder latency model
    p.add_argument("--dim", type=int, default=768, help="Embedding dimension.")
    p.add_argument("--request-overhead-ms", type=float, default=60.0, help="Fixed overhead per embedding request (ms).")
    p.add_argument("--per-item-ms", type=float, default=1.2, help="Per-item cost inside embedding request (ms).")

    # Chroma
    p.add_argument("--persist-dir", type=str, default="", help="Optional Chroma persistence dir (empty = in-memory).")
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    persist_dir = args.persist_dir.strip() or None

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        write_synthetic_corpus(
            data_dir=data_dir,
            nodes=args.nodes,
            node_chars=args.node_chars,
            num_files=args.files,
            seed=args.seed,
        )

        embedder = FakeEmbedder(
            dim=args.dim,
            request_overhead_ms=args.request_overhead_ms,
            per_item_ms=args.per_item_ms,
        )

        rows: List[ResultRow] = []

        # ---------------- Set 1 ----------------
        t0 = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = transform_sync(docs, num_workers=None)
        collection = init_chroma(persist_dir, "bench_set1")
        embed_s, upsert_s = await embed_and_upsert(
            nodes=nodes,
            embedder=embedder,
            collection=collection,
            embed_batch_size=1,
            upsert_batch_size=1,
            embed_num_workers=1,  # no concurrency to match "no workers anywhere"
        )
        total_s = time.perf_counter() - t0
        rows.append(ResultRow("Set1_Default", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------------- Set 2 ----------------
        t0 = time.perf_counter()
        docs, load_s = load_docs_parallel(data_dir, num_workers=args.reader_workers)
        nodes, transform_s = transform_sync(docs, num_workers=None)
        collection = init_chroma(persist_dir, "bench_set2")
        embed_s, upsert_s = await embed_and_upsert(
            nodes=nodes,
            embedder=embedder,
            collection=collection,
            embed_batch_size=1,
            upsert_batch_size=1,
            embed_num_workers=1,  # keep "pipeline still sync" and no extra embed concurrency
        )
        total_s = time.perf_counter() - t0
        rows.append(ResultRow("Set2_ReaderParallel", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------------- Set 3 ----------------
        t0 = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = transform_sync(docs, num_workers=args.pipeline_workers)
        collection = init_chroma(persist_dir, "bench_set3")
        embed_s, upsert_s = await embed_and_upsert(
            nodes=nodes,
            embedder=embedder,
            collection=collection,
            embed_batch_size=1,
            upsert_batch_size=1,
            embed_num_workers=1,  # keep embedding/upsert sequential to isolate pipeline parallelism
        )
        total_s = time.perf_counter() - t0
        rows.append(ResultRow("Set3_PipelineParallelSync", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------------- Set 4 ----------------
        t0 = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes_async, transform_s = await transform_async(docs, num_workers=args.async_workers)
        collection = init_chroma(persist_dir, "bench_set4")
        embed_s, upsert_s = await embed_and_upsert(
            nodes=nodes_async,
            embedder=embedder,
            collection=collection,
            embed_batch_size=1,   # key: no batching
            upsert_batch_size=1,  # key: no batching
            embed_num_workers=args.async_workers,  # concurrency only
        )
        total_s = time.perf_counter() - t0
        rows.append(ResultRow("Set4_AsyncOnly", len(nodes_async), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------------- Set 5 ----------------
        t0 = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes_async, transform_s = await transform_async(docs, num_workers=args.async_workers)
        collection = init_chroma(persist_dir, "bench_set5")
        embed_s, upsert_s = await embed_and_upsert(
            nodes=nodes_async,
            embedder=embedder,
            collection=collection,
            embed_batch_size=max(1, args.set5_embed_batch),
            upsert_batch_size=max(1, args.set5_upsert_batch),
            embed_num_workers=args.async_workers,
        )
        total_s = time.perf_counter() - t0
        rows.append(ResultRow("Set5_AsyncPlusBatching", len(nodes_async), load_s, transform_s, embed_s, upsert_s, total_s))

        print_table(rows)

        # Small highlight: Set5 vs Set4
        r4 = next(r for r in rows if r.config == "Set4_AsyncOnly")
        r5 = next(r for r in rows if r.config == "Set5_AsyncPlusBatching")
        print(f"Set5 vs Set4 total improvement: {pct_faster(r4.total_s, r5.total_s):.1f}% faster\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

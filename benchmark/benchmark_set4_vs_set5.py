#!/usr/bin/env python3
"""
Benchmark Set 4 vs Set 5 for an ingestion-like pipeline.
Author: Arup Sarker, djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 28/12/2025

Set 4: Async concurrency only
  - embed_batch_size = 1 (one-by-one embedding calls)
  - upsert_batch_size = 1 (one-by-one vector DB inserts)
  - async concurrency via semaphore (num_workers)

Set 5: Async + batching
  - embed_batch_size = configurable (e.g., 64)
  - upsert_batch_size = configurable (e.g., 256)
  - same async concurrency via semaphore (num_workers)

Vector DB: ChromaDB (local)
Embeddings: FakeEmbedder that simulates API-like latency:
  latency = request_overhead_ms + per_item_ms * batch_size

This demonstrates the core performance argument:
  batching reduces per-request overhead and improves throughput,
  especially as node count grows.

Usage example:
  python benchmark_set4_vs_set5.py --nodes 5000 --dim 768 --num-workers 32 \
    --set5-embed-batch 64 --set5-upsert-batch 256 \
    --request-overhead-ms 60 --per-item-ms 1.2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import random
import string
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Dict, Any

import chromadb


# -------------------------
# Synthetic Node Generator
# -------------------------

@dataclass
class Node:
    node_id: str
    text: str
    metadata: Dict[str, Any]


def _rand_text(chars: int, seed: int) -> str:
    rnd = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + "     "
    return "".join(rnd.choice(alphabet) for _ in range(chars)).strip()


def make_nodes(num_nodes: int, text_chars: int, seed: int) -> List[Node]:
    nodes: List[Node] = []
    for i in range(num_nodes):
        nid = f"node-{i}"
        txt = _rand_text(text_chars, seed + i)
        nodes.append(Node(node_id=nid, text=txt, metadata={"i": i}))
    return nodes


# -------------------------
# Fake Embedder (API-like)
# -------------------------

class FakeEmbedder:
    """
    Simulates a remote embedding API:
      total_latency(batch) = request_overhead_ms + per_item_ms * len(batch)
    Returns deterministic vectors of `dim` floats in [-1, 1] based on hash(text).
    """
    def __init__(self, dim: int, request_overhead_ms: float, per_item_ms: float):
        self.dim = dim
        self.request_overhead_ms = request_overhead_ms
        self.per_item_ms = per_item_ms

    async def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        # Simulate API cost
        sleep_s = (self.request_overhead_ms + self.per_item_ms * len(texts)) / 1000.0
        await asyncio.sleep(sleep_s)

        # Deterministic vectors (fast enough for benchmarking)
        out: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Expand hash to dim floats deterministically
            vec = []
            for j in range(self.dim):
                b = h[j % len(h)]
                # Map [0,255] -> [-1,1]
                vec.append((b / 127.5) - 1.0)
            out.append(vec)
        return out


# -------------------------
# Chroma Helpers
# -------------------------

def init_chroma(persist_dir: str | None, collection_name: str) -> Tuple[Any, Any]:
    """
    Returns (client, collection).
    If persist_dir is provided, the DB is persisted there.
    """
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.EphemeralClient()

    # Recreate collection fresh each run
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
    return client, collection


async def chroma_upsert_batches(
    collection: Any,
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Sequence[Dict[str, Any]],
    documents: Sequence[str],
    upsert_batch_size: int,
) -> None:
    """
    Chroma insert in batches.
    """
    total = len(ids)
    for start in range(0, total, upsert_batch_size):
        end = min(total, start + upsert_batch_size)
        collection.add(
            ids=list(ids[start:end]),
            embeddings=[list(x) for x in embeddings[start:end]],
            metadatas=list(metadatas[start:end]),
            documents=list(documents[start:end]),
        )
        # Yield control to event loop (helps when batch size is small)
        await asyncio.sleep(0)


# -------------------------
# Pipeline-style Runner
# -------------------------

@dataclass
class RunResult:
    name: str
    nodes: int
    embed_batch_size: int
    upsert_batch_size: int
    num_workers: int
    embed_time_s: float
    upsert_time_s: float
    total_time_s: float


async def embed_all_async(
    embedder: FakeEmbedder,
    texts: Sequence[str],
    embed_batch_size: int,
    num_workers: int,
) -> List[List[float]]:
    """
    Async embedding with concurrency limit and batching.
    """
    sem = asyncio.Semaphore(num_workers)

    async def _embed_one_batch(batch_texts: Sequence[str]) -> List[List[float]]:
        async with sem:
            return await embedder.embed_batch(batch_texts)

    tasks = []
    for start in range(0, len(texts), embed_batch_size):
        batch = texts[start:start + embed_batch_size]
        tasks.append(asyncio.create_task(_embed_one_batch(batch)))

    # Preserve order
    batch_results = await asyncio.gather(*tasks)
    out: List[List[float]] = []
    for br in batch_results:
        out.extend(br)
    return out


async def run_config(
    name: str,
    nodes: List[Node],
    embedder: FakeEmbedder,
    persist_dir: str | None,
    collection_name: str,
    embed_batch_size: int,
    upsert_batch_size: int,
    num_workers: int,
) -> RunResult:
    _, collection = init_chroma(persist_dir, collection_name)

    texts = [n.text for n in nodes]
    ids = [n.node_id for n in nodes]
    metadatas = [n.metadata for n in nodes]

    t0 = time.perf_counter()
    te0 = time.perf_counter()
    embs = await embed_all_async(
        embedder=embedder,
        texts=texts,
        embed_batch_size=embed_batch_size,
        num_workers=num_workers,
    )
    te1 = time.perf_counter()

    tu0 = time.perf_counter()
    await chroma_upsert_batches(
        collection=collection,
        ids=ids,
        embeddings=embs,
        metadatas=metadatas,
        documents=texts,
        upsert_batch_size=upsert_batch_size,
    )
    tu1 = time.perf_counter()

    t1 = time.perf_counter()

    return RunResult(
        name=name,
        nodes=len(nodes),
        embed_batch_size=embed_batch_size,
        upsert_batch_size=upsert_batch_size,
        num_workers=num_workers,
        embed_time_s=te1 - te0,
        upsert_time_s=tu1 - tu0,
        total_time_s=t1 - t0,
    )


def pct_improvement(baseline: float, new: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - new) / baseline * 100.0


# -------------------------
# CLI / Main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Set4 vs Set5 (Async vs Async+Batching) using Chroma.")
    p.add_argument("--nodes", type=int, default=5000, help="Total number of nodes to ingest.")
    p.add_argument("--text-chars", type=int, default=1200, help="Synthetic text length per node (chars).")
    p.add_argument("--seed", type=int, default=7, help="Seed for synthetic data generation.")
    p.add_argument("--dim", type=int, default=768, help="Embedding dimension.")
    p.add_argument("--num-workers", type=int, default=32, help="Async concurrency limit (semaphore).")

    # Fake embedder latency model
    p.add_argument("--request-overhead-ms", type=float, default=60.0,
                   help="Simulated fixed overhead per embedding request (ms).")
    p.add_argument("--per-item-ms", type=float, default=1.2,
                   help="Simulated per-item cost inside a request (ms per text).")

    # Chroma persistence
    p.add_argument("--persist-dir", type=str, default="",
                   help="If set, Chroma persists here. If empty, uses in-memory Chroma.")
    p.add_argument("--collection", type=str, default="bench_set4_set5", help="Chroma collection name.")

    # Set 4 vs Set 5 batching
    p.add_argument("--set4-embed-batch", type=int, default=1, help="Set4 embedding batch size (default 1).")
    p.add_argument("--set4-upsert-batch", type=int, default=1, help="Set4 upsert batch size (default 1).")

    p.add_argument("--set5-embed-batch", type=int, default=64, help="Set5 embedding batch size.")
    p.add_argument("--set5-upsert-batch", type=int, default=256, help="Set5 upsert batch size.")

    return p.parse_args()


async def main_async() -> None:
    args = parse_args()

    persist_dir = args.persist_dir.strip() or None

    nodes = make_nodes(num_nodes=args.nodes, text_chars=args.text_chars, seed=args.seed)
    embedder = FakeEmbedder(
        dim=args.dim,
        request_overhead_ms=args.request_overhead_ms,
        per_item_ms=args.per_item_ms,
    )

    # SET 4
    r4 = await run_config(
        name="Set4_AsyncOnly",
        nodes=nodes,
        embedder=embedder,
        persist_dir=persist_dir,
        collection_name=args.collection + "_set4",
        embed_batch_size=args.set4_embed_batch,
        upsert_batch_size=args.set4_upsert_batch,
        num_workers=args.num_workers,
    )

    # SET 5
    r5 = await run_config(
        name="Set5_AsyncPlusBatching",
        nodes=nodes,
        embedder=embedder,
        persist_dir=persist_dir,
        collection_name=args.collection + "_set5",
        embed_batch_size=args.set5_embed_batch,
        upsert_batch_size=args.set5_upsert_batch,
        num_workers=args.num_workers,
    )

    # Report
    print("\n=== RESULTS ===")
    for r in (r4, r5):
        print(
            f"{r.name}: nodes={r.nodes}, num_workers={r.num_workers}, "
            f"embed_batch={r.embed_batch_size}, upsert_batch={r.upsert_batch_size}\n"
            f"  embed_time  = {r.embed_time_s:.3f}s\n"
            f"  upsert_time = {r.upsert_time_s:.3f}s\n"
            f"  total_time  = {r.total_time_s:.3f}s\n"
        )

    print("=== IMPROVEMENT (Set5 vs Set4) ===")
    print(f"Embedding stage: {pct_improvement(r4.embed_time_s, r5.embed_time_s):.2f}% faster")
    print(f"Upsert stage:    {pct_improvement(r4.upsert_time_s, r5.upsert_time_s):.2f}% faster")
    print(f"Total pipeline:  {pct_improvement(r4.total_time_s, r5.total_time_s):.2f}% faster\n")

    # Small sanity check: expected trend
    if r5.total_time_s > r4.total_time_s:
        print("Note: Set5 was slower in this run. Likely causes:")
        print("- Very low request_overhead_ms (batching helps less when overhead is tiny)")
        print("- embed_batch/upsert_batch too large (memory/DB overhead dominates)")
        print("- num_workers too low to keep pipeline busy\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

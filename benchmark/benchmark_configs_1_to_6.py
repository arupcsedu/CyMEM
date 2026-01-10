#!/usr/bin/env python3
#Author: Arup Sarker, djy8hg@virginia.edu, arupcsedu@gmail.com
# Date: 07/01/2026
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
from typing import Any, Dict, List, Sequence, Tuple, Optional

# ---- LlamaIndex ----
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TextNode, TransformComponent

# ---- Optional Chroma ----
try:
    import chromadb
except Exception:
    chromadb = None

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
# Transform: delimiter splitter (must be TransformComponent)
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

    async def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        sleep_s = (self.request_overhead_ms + self.per_item_ms * len(texts)) / 1000.0
        await asyncio.sleep(sleep_s)

        out: List[List[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = []
            for j in range(self.dim):
                b = h[j % len(h)]
                vec.append((b / 127.5) - 1.0)
            out.append(vec)
        return out


# -------------------------
# Vector Stores
# -------------------------

class ThreadSafeMemVectorStore:
    """Thread-safe in-memory store to allow true parallel upserts."""
    def __init__(self):
        self._lock = asyncio.Lock()
        self._data: Dict[str, Tuple[List[float], Dict[str, Any], str]] = {}

    async def add_batch(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        documents: Sequence[str],
    ) -> None:
        async with self._lock:
            for i in range(len(ids)):
                self._data[ids[i]] = (list(embeddings[i]), dict(metadatas[i]), documents[i])


def init_chroma_collection(persist_dir: Optional[str], name: str):
    if chromadb is None:
        raise RuntimeError("chromadb not installed. Install or use --vector-store mem.")
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_dir)
    else:
        client = chromadb.EphemeralClient()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


async def chroma_add_batch(collection, ids, embeddings, metadatas, documents) -> None:
    # Chroma add is synchronous; call in a thread to avoid blocking event loop
    def _do():
        collection.add(
            ids=list(ids),
            embeddings=[list(x) for x in embeddings],
            metadatas=list(metadatas),
            documents=list(documents),
        )
    await asyncio.to_thread(_do)


# -------------------------
# Common async embedding runner
# -------------------------

async def embed_all_async(embedder: FakeEmbedder, texts: Sequence[str], batch_size: int, num_workers: int):
    sem = asyncio.Semaphore(num_workers)

    async def _run(batch: Sequence[str]):
        async with sem:
            return await embedder.embed_batch(batch)

    tasks = []
    for s in range(0, len(texts), batch_size):
        tasks.append(asyncio.create_task(_run(texts[s:s + batch_size])))

    batches = await asyncio.gather(*tasks)
    out = []
    for b in batches:
        out.extend(b)
    return out


def chunk_list(n: int, batch_size: int) -> List[Tuple[int, int]]:
    return [(i, min(n, i + batch_size)) for i in range(0, n, batch_size)]


# -------------------------
# Benchmark result structure
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
    return (baseline - new) / baseline * 100.0 if baseline > 0 else 0.0


def print_table(rows: List[ResultRow]) -> None:
    headers = ["Config", "Nodes", "Load(s)", "Transform(s)", "Embed(s)", "Upsert(s)", "Total(s)", "Δ vs Set1"]
    base = rows[0].total_s if rows else 0.0

    data = []
    for r in rows:
        delta = "baseline" if r.config == "Set1_Default" else f"{pct_faster(base, r.total_s):.1f}% faster"
        data.append([
            r.config, str(r.nodes),
            f"{r.load_s:.3f}", f"{r.transform_s:.3f}",
            f"{r.embed_s:.3f}", f"{r.upsert_s:.3f}",
            f"{r.total_s:.3f}", delta
        ])

    cols = list(zip(headers, *data))
    widths = [max(len(x) for x in col) for col in cols]

    def fmt(items):
        return " | ".join(items[i].ljust(widths[i]) for i in range(len(items)))

    sep = "-+-".join("-" * w for w in widths)

    print("\n" + fmt(headers))
    print(sep)
    for row in data:
        print(fmt(row))
    print()


# -------------------------
# Load + transform helpers
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
# Upsert implementations
# -------------------------

async def upsert_sequential(
    store_kind: str,
    store_obj: Any,
    ids, embs, metas, docs,
    upsert_batch_size: int,
) -> float:
    t0 = time.perf_counter()
    n = len(ids)
    for s, e in chunk_list(n, upsert_batch_size):
        if store_kind == "mem":
            await store_obj.add_batch(ids[s:e], embs[s:e], metas[s:e], docs[s:e])
        else:
            await chroma_add_batch(store_obj, ids[s:e], embs[s:e], metas[s:e], docs[s:e])
    return time.perf_counter() - t0


async def upsert_parallel_bsp(
    store_kind: str,
    store_obj: Any,
    ids, embs, metas, docs,
    upsert_batch_size: int,
    upsert_workers: int,
) -> float:
    """
    BSP-style parallel upsert:
      - Partition into batches
      - Launch batch upserts concurrently (bounded)
      - Barrier = await gather()
    """
    t0 = time.perf_counter()
    sem = asyncio.Semaphore(upsert_workers)
    n = len(ids)

    async def _one_batch(s: int, e: int):
        async with sem:
            if store_kind == "mem":
                await store_obj.add_batch(ids[s:e], embs[s:e], metas[s:e], docs[s:e])
            else:
                await chroma_add_batch(store_obj, ids[s:e], embs[s:e], metas[s:e], docs[s:e])

    tasks = [asyncio.create_task(_one_batch(s, e)) for s, e in chunk_list(n, upsert_batch_size)]
    await asyncio.gather(*tasks)  # barrier
    return time.perf_counter() - t0


# -------------------------
# Main benchmark
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=20000)
    p.add_argument("--node-chars", type=int, default=800)
    p.add_argument("--files", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--reader-workers", type=int, default=8)
    p.add_argument("--pipeline-workers", type=int, default=8)
    p.add_argument("--async-workers", type=int, default=32)

    p.add_argument("--set5-embed-batch", type=int, default=64)
    p.add_argument("--set5-upsert-batch", type=int, default=256)

    # BSP knobs (Set6)
    p.add_argument("--bsp-embed-batch", type=int, default=128)
    p.add_argument("--bsp-upsert-batch", type=int, default=512)
    p.add_argument("--bsp-upsert-workers", type=int, default=16)

    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--request-overhead-ms", type=float, default=60.0)
    p.add_argument("--per-item-ms", type=float, default=1.2)

    p.add_argument("--vector-store", choices=["chroma", "mem"], default="chroma")
    p.add_argument("--persist-dir", type=str, default="")
    return p.parse_args()


async def main_async():
    args = parse_args()
    persist_dir = args.persist_dir.strip() or None

    embedder = FakeEmbedder(args.dim, args.request_overhead_ms, args.per_item_ms)

    rows: List[ResultRow] = []

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        write_synthetic_corpus(data_dir, args.nodes, args.node_chars, args.files, args.seed)

        def make_store(name: str):
            if args.vector_store == "mem":
                return "mem", ThreadSafeMemVectorStore()
            return "chroma", init_chroma_collection(persist_dir, name)

        # ---------- Set 1 ----------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = transform_sync(docs, num_workers=None)

        sk, store = make_store("set1")
        texts = [n.text for n in nodes]
        ids = [n.id_ for n in nodes]
        metas = [dict(n.metadata or {}) for n in nodes]

        t0 = time.perf_counter()
        embs = await embed_all_async(embedder, texts, batch_size=1, num_workers=1)
        embed_s = time.perf_counter() - t0

        upsert_s = await upsert_sequential(sk, store, ids, embs, metas, texts, upsert_batch_size=1)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set1_Default", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------- Set 2 ----------
        t_all = time.perf_counter()
        docs, load_s = load_docs_parallel(data_dir, args.reader_workers)
        nodes, transform_s = transform_sync(docs, num_workers=None)

        sk, store = make_store("set2")
        texts = [n.text for n in nodes]
        ids = [n.id_ for n in nodes]
        metas = [dict(n.metadata or {}) for n in nodes]

        t0 = time.perf_counter()
        embs = await embed_all_async(embedder, texts, batch_size=1, num_workers=1)
        embed_s = time.perf_counter() - t0

        upsert_s = await upsert_sequential(sk, store, ids, embs, metas, texts, upsert_batch_size=1)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set2_ReaderParallel", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------- Set 3 ----------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = transform_sync(docs, num_workers=args.pipeline_workers)

        sk, store = make_store("set3")
        texts = [n.text for n in nodes]
        ids = [n.id_ for n in nodes]
        metas = [dict(n.metadata or {}) for n in nodes]

        t0 = time.perf_counter()
        embs = await embed_all_async(embedder, texts, batch_size=1, num_workers=1)
        embed_s = time.perf_counter() - t0

        upsert_s = await upsert_sequential(sk, store, ids, embs, metas, texts, upsert_batch_size=1)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set3_PipelineParallelSync", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------- Set 4 ----------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)

        sk, store = make_store("set4")
        texts = [n.text for n in nodes]
        ids = [n.id_ for n in nodes]
        metas = [dict(n.metadata or {}) for n in nodes]

        t0 = time.perf_counter()
        embs = await embed_all_async(embedder, texts, batch_size=1, num_workers=args.async_workers)
        embed_s = time.perf_counter() - t0

        upsert_s = await upsert_sequential(sk, store, ids, embs, metas, texts, upsert_batch_size=1)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set4_AsyncOnly", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------- Set 5 ----------
        t_all = time.perf_counter()
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)

        sk, store = make_store("set5")
        texts = [n.text for n in nodes]
        ids = [n.id_ for n in nodes]
        metas = [dict(n.metadata or {}) for n in nodes]

        t0 = time.perf_counter()
        embs = await embed_all_async(embedder, texts, batch_size=args.set5_embed_batch, num_workers=args.async_workers)
        embed_s = time.perf_counter() - t0

        upsert_s = await upsert_sequential(sk, store, ids, embs, metas, texts, upsert_batch_size=args.set5_upsert_batch)
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set5_AsyncPlusBatching", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        # ---------- Set 6 (BSP) ----------
        # BSP supersteps:
        #   A) load + transform  -> barrier
        #   B) embed (parallel batches) -> barrier
        #   C) upsert (parallel batches) -> barrier
        t_all = time.perf_counter()

        # Superstep A
        docs, load_s = load_docs_sync(data_dir)
        nodes, transform_s = await transform_async(docs, args.async_workers)
        # barrier implicit

        texts = [n.text for n in nodes]
        ids = [n.id_ for n in nodes]
        metas = [dict(n.metadata or {}) for n in nodes]

        # Superstep B: parallel embedding batches + barrier
        t0 = time.perf_counter()
        embs = await embed_all_async(embedder, texts, batch_size=args.bsp_embed_batch, num_workers=args.async_workers)
        embed_s = time.perf_counter() - t0
        # barrier implicit (await finished)

        # Superstep C: parallel upsert batches + barrier
        sk, store = make_store("set6")
        upsert_s = await upsert_parallel_bsp(
            sk, store, ids, embs, metas, texts,
            upsert_batch_size=args.bsp_upsert_batch,
            upsert_workers=args.bsp_upsert_workers,
        )
        total_s = time.perf_counter() - t_all
        rows.append(ResultRow("Set6_BSP_BulkSyncParallel", len(nodes), load_s, transform_s, embed_s, upsert_s, total_s))

        print_table(rows)

        r4 = next(r for r in rows if r.config == "Set4_AsyncOnly")
        r5 = next(r for r in rows if r.config == "Set5_AsyncPlusBatching")
        r6 = next(r for r in rows if r.config == "Set6_BSP_BulkSyncParallel")
        print(f"Set5 vs Set4: {pct_faster(r4.total_s, r5.total_s):.1f}% faster")
        print(f"Set6 vs Set5: {pct_faster(r5.total_s, r6.total_s):.1f}% faster")
        print(f"Set6 vs Set4: {pct_faster(r4.total_s, r6.total_s):.1f}% faster\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

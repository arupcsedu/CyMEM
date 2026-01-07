"""
main_baseline_v2.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 21/12/2025

Baseline (Replicated Build) that mirrors the working global run behavior,
but without importing main.py or HuggingFace datasets.

Each rank does:
  - preprocess (its shard, depending on preprocessing.py rank-awareness)
  - embed corpus chunks
  - build local vectorstore (redundant across ranks)
  - eval on a shard of queries from --eval_dir (JSON files)

Outputs:
  - --lat_csv / --lat_json: aggregate per-rank file writing if provided,
    otherwise writes runs/<tag>/latencies_rank{rank}.csv/json
  - --llm_stats_json: optional, per-rank write if provided (or auto)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import get_hpc_shard
from metrics import metrics, record_latency

from preprocessing import preprocess_documents
from embedder import Embedder, EmbedderConfig
from vectorstore import VectorStore, VectorStoreConfig
from memory import MemoryConfig, MemoryModule
from agents import RagAgent, AgentConfig, LLMConfig, LLMGenerator

logger = logging.getLogger(__name__)


def setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s:%(name)s:%(message)s",
    )


def resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data/indexing
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--max_chars", type=int, default=1000)
    p.add_argument("--overlap_chars", type=int, default=200)
    p.add_argument("--file_glob", type=str, default="*")

    # Embedder
    p.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    # Vectorstore
    p.add_argument("--use_faiss", action="store_true")
    p.add_argument("--faiss_index_type", type=str, default="IndexFlatL2")
    p.add_argument("--normalize_embeddings", action="store_true")

    # LLM
    p.add_argument("--llm", type=str, default="gpt2")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)

    # Eval
    p.add_argument("--eval_dir", type=str, default=None)
    p.add_argument("--eval_max_samples", type=int, default=None)

    # Outputs (match global style)
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--tag", type=str, default="baseline")
    p.add_argument("--lat_csv", type=str, default=None)
    p.add_argument("--lat_json", type=str, default=None)
    p.add_argument("--llm_stats_json", type=str, default=None)

    return p.parse_args()


def load_queries(eval_dir: str) -> List[str]:
    """
    Read eval_dir/*.json. Each file can be:
      - {"query": "..."}
      - [{"query": "..."}, ...]
    Returns list of query strings.
    """
    paths = sorted(glob.glob(os.path.join(eval_dir, "*.json")))
    logger.info("Eval dir=%s, found %d json file(s).", eval_dir, len(paths))

    queries: List[str] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Skipping unreadable json: %s (%s)", p, e)
            continue

        if isinstance(data, dict) and "query" in data:
            queries.append(str(data["query"]))
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict) and "query" in rec:
                    queries.append(str(rec["query"]))

    logger.info("Loaded %d total query record(s) from %s", len(queries), eval_dir)
    return queries


def shard_list(items: List[Any], rank: int, world_size: int) -> List[Any]:
    if world_size <= 1:
        return items
    return [x for i, x in enumerate(items) if (i % world_size) == rank]


def build_agent(args: argparse.Namespace) -> RagAgent:
    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    # Preprocess
    with record_latency("baseline.preprocess_total", store_samples=True):
        chunks, metadatas = preprocess_documents(
            input_path=args.data_dir,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            file_glob=args.file_glob,
        )
    if not chunks:
        raise RuntimeError("No chunks produced from --data_dir (baseline).")

    # Embed
    embedder = Embedder(EmbedderConfig(model_name=args.encoder, device=device))
    with record_latency("baseline.embed_total", store_samples=True):
        emb = embedder.embed_corpus(chunks)
    dim = int(emb.shape[1])
    logger.info("Embedding dim=%d", dim)

    # Local vectorstore (redundant baseline)
    vs = VectorStore(VectorStoreConfig(
        dim=dim,
        use_faiss=bool(args.use_faiss),
        faiss_index_type=args.faiss_index_type,
        normalize_embeddings=bool(args.normalize_embeddings),
    ))
    with record_latency("baseline.index_total", store_samples=True):
        vs.add_documents(embeddings=emb, texts=chunks, metadatas=metadatas)
    # memory + llm + agent
    '''
    import inspect

    def build_memory_module(dim: int) -> MemoryModule:
        sig = inspect.signature(MemoryModule.__init__)
        params = sig.parameters
        if "dim" in params:
            return MemoryModule(dim=dim)
        if "embedding_dim" in params:
            return MemoryModule(embedding_dim=dim)
        if "d" in params:
            return MemoryModule(d=dim)
        return MemoryModule()

    mem = build_memory_module(dim)
    '''
    mem = MemoryModule(MemoryConfig(dim=dim))
    llm = LLMGenerator(LLMConfig(
        model_name=args.llm,
        device=device,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    ))

    return RagAgent(embedder=embedder, vectorstore=vs, memory=mem, llm=llm, config=AgentConfig())


def run_eval(agent: RagAgent, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run eval on rank shard. Returns LLM generation stats if available in agent/llm.
    """
    rank, world_size = get_hpc_shard()
    if not args.eval_dir:
        logger.info("No --eval_dir provided; skipping eval.")
        return {}

    queries = load_queries(args.eval_dir)
    if args.eval_max_samples is not None:
        queries = queries[: args.eval_max_samples]

    local_queries = shard_list(queries, rank, world_size)
    logger.info("Rank %d/%d evaluating %d/%d queries", rank, world_size, len(local_queries), len(queries))

    with record_latency("baseline.eval_total", store_samples=True):
        for q in local_queries:
            with record_latency("baseline.eval_one", store_samples=True):
                _ans, _dbg = agent.generate_answer(q)

    # If your LLMGenerator stores stats somewhere else, adjust here.
    # Many repos store stats in metrics; we keep it empty unless you implemented it.
    return {}


def write_outputs(args: argparse.Namespace, rank: int) -> None:
    out_root = Path(args.out_dir) / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Default per-rank
    lat_json = args.lat_json or str(out_root / f"latencies_rank{rank}.json")
    lat_csv = args.lat_csv or str(out_root / f"latencies_rank{rank}.csv")

    metrics.dump_json(lat_json)
    metrics.dump_csv(lat_csv)

    logger.info("Wrote latencies: %s and %s", lat_json, lat_csv)


def main() -> None:
    args = parse_args()
    rank, world_size = get_hpc_shard()
    setup_logging(rank)

    logger.info("HPC shard: rank=%d world_size=%d", rank, world_size)
    logger.info("BASELINE v2: every rank builds local index (redundant).")

    with record_latency("baseline.pipeline_total_run", store_samples=True):
        agent = build_agent(args)
        _stats = run_eval(agent, args)

    write_outputs(args, rank)
    logger.info("Baseline v2 finished on rank=%d", rank)


if __name__ == "__main__":
    main()

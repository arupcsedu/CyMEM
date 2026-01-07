"""
main_baseline.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 18/12/2025

Baseline entrypoint for Deep RC RAG (replicated build):
  - Every rank preprocesses, embeds, builds its own vectorstore (redundant).
  - Evaluation reads ONLY local JSON files (no HuggingFace datasets).
  - Does NOT import main.py (avoids datasets->pandas GLIBCXX crash).
  - Does NOT modify metrics.py.

Expected eval format:
  eval_dir/*.json where each file is either:
    {"query": "..."} or [{"query":"..."}, ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import get_hpc_shard
from metrics import (
    metrics,
    record_latency,
    summarize_throughput,
    export_throughput_json,
    export_throughput_csv,
)

from preprocessing import preprocess_documents
from embedder import Embedder, EmbedderConfig
from vectorstore import VectorStore, VectorStoreConfig
from memory import MemoryConfig, MemoryModule
from agents import RagAgent, AgentConfig, LLMConfig, LLMGenerator

logger = logging.getLogger(__name__)


def _setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s:%(name)s:%(message)s",
    )


def _resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Corpus/indexing
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--max_chars", type=int, default=1000)
    p.add_argument("--overlap_chars", type=int, default=200)
    p.add_argument("--file_glob", type=str, default="*")

    # Embeddings
    p.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    # Vector store
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

    # Eval (local only, JSON files)
    p.add_argument("--eval_dir", type=str, default=None)
    p.add_argument("--eval_max_samples", type=int, default=None)
    p.add_argument("--eval_out_dir", type=str, default=None, help="Optional per-rank eval outputs dir")

    # Outputs
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--tag", type=str, default="baseline")

    return p.parse_args()


def run_eval_from_local_files(
    agent: RagAgent,
    eval_dir: str,
    max_samples: Optional[int] = None,
    eval_out_dir: Optional[str] = None,
) -> None:
    """
    Rank-aware eval:
      - loads all queries from eval_dir/*.json
      - shards by (i % world_size == rank)
      - writes per-rank jsonl outputs (optional)
    """
    rank, world_size = get_hpc_shard()

    paths = sorted(glob.glob(f"{eval_dir}/*.json"))
    queries: List[Dict[str, Any]] = []

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "query" in data:
            queries.append({"query": str(data["query"]), "source": p})
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict) and "query" in rec:
                    queries.append({"query": str(rec["query"]), "source": p})

    if max_samples is not None:
        queries = queries[:max_samples]

    local = [(i, q) for i, q in enumerate(queries) if i % max(world_size, 1) == rank]

    logger.info(
        "Rank %d/%d evaluating %d of %d queries from %s",
        rank, world_size, len(local), len(queries), eval_dir,
    )

    out_f = None
    if eval_out_dir:
        out_path = Path(eval_out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path / f"eval_outputs_rank{rank}.jsonl", "w", encoding="utf-8")

    with record_latency("eval_total_run", store_samples=True):
        for i, rec in local:
            q = rec["query"]
            with record_latency("eval_one_query", store_samples=True):
                ans, dbg = agent.generate_answer(q)

            if out_f:
                out_f.write(json.dumps({
                    "idx": i,
                    "query": q,
                    "answer": ans,
                    "source": rec.get("source"),
                }) + "\n")

    if out_f:
        out_f.close()


def build_pipeline_baseline(args: argparse.Namespace) -> Tuple[RagAgent, int]:
    """
    Baseline pipeline on each rank (redundant):
      preprocess -> embed -> local vectorstore build -> agent
    """
    device = _resolve_device(args.device)
    logger.info("Using device: %s", device)

    with record_latency("baseline.pipeline_total_run", store_samples=True):

        # preprocess
        logger.info("Preprocessing corpus from %s", args.data_dir)
        with record_latency("baseline.preprocess_total", store_samples=True):
            chunks, metadatas = preprocess_documents(
                input_path=args.data_dir,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
                file_glob=args.file_glob,
            )
        if not chunks:
            raise RuntimeError("No chunks produced. Check --data_dir and file types.")

        # embed
        embedder = Embedder(EmbedderConfig(model_name=args.encoder, device=device))
        logger.info("Embedding %d corpus chunks...", len(chunks))
        with record_latency("baseline.embed_total", store_samples=True):
            embeddings = embedder.embed_corpus(chunks)

        dim = int(embeddings.shape[1])
        logger.info("Embedding dim=%d", dim)

        # vectorstore
        vs_cfg = VectorStoreConfig(
            dim=dim,
            use_faiss=bool(args.use_faiss),
            faiss_index_type=args.faiss_index_type,
            normalize_embeddings=bool(args.normalize_embeddings),
        )
        vectorstore = VectorStore(vs_cfg)

        logger.info("Building LOCAL vectorstore on this rank (baseline)...")
        with record_latency("baseline.index_total", store_samples=True):
            vectorstore.add_documents(
                embeddings=embeddings,
                texts=chunks,
                metadatas=metadatas,
            )

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

        memory = build_memory_module(dim)
        '''
        memory = MemoryModule(MemoryConfig(dim=dim))

        llm = LLMGenerator(
            LLMConfig(
                model_name=args.llm,
                device=device,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )
        agent = RagAgent(
            embedder=embedder,
            vectorstore=vectorstore,
            memory=memory,
            llm=llm,
            config=AgentConfig(),
        )

    return agent, dim


def write_outputs(args: argparse.Namespace, rank: int) -> None:
    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics.dump_json(str(out_dir / f"latencies_rank{rank}.json"))
    metrics.dump_csv(str(out_dir / f"latencies_rank{rank}.csv"))

    # optional throughput exports
    try:
        export_throughput_json(metrics, out_dir / f"throughput_rank{rank}.json")
        export_throughput_csv(metrics, out_dir / f"throughput_rank{rank}.csv")
    except Exception as e:
        logger.warning("Throughput export skipped: %s", e)

    logger.info("Wrote baseline outputs to %s", out_dir)


def main() -> None:
    args = parse_args()
    rank, world_size = get_hpc_shard()
    _setup_logging(rank)

    logger.info("HPC shard detected: rank=%d world_size=%d", rank, world_size)
    logger.info("BASELINE: each rank preprocess+embed+index (redundant)")

    agent, dim = build_pipeline_baseline(args)

    if args.eval_dir:
        run_eval_from_local_files(
            agent=agent,
            eval_dir=args.eval_dir,
            max_samples=args.eval_max_samples,
            eval_out_dir=args.eval_out_dir,
        )

    # optional pretty throughput print
    try:
        summarize_throughput(metrics)
    except Exception as e:
        logger.warning("summarize_throughput skipped: %s", e)

    write_outputs(args, rank)
    logger.info("Baseline rank %d complete.", rank)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
main.py

Deep RC RAG + Agentic Memory demo.
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 09/15/2025

Key features:
  - Builds a RAG pipeline from a local corpus directory (preprocessing -> embedding -> vectorstore)
  - Runs evaluation either:
      (A) from local files in --eval_dir (txt/md/log/json/jsonl)
      (B) from Hugging Face dataset (wikitext by default)
  - HPC/Slurm-friendly:
      * Each rank runs the same code
      * Evaluation is deterministically sharded by FILE across ranks
      * Logs show rank/world_size and which files were assigned

Metrics:
  - Your existing module-level latencies are recorded via metrics.py
  - context_build_total is recorded inside agents.py
  - llm.generate recorded inside agents.py LLMGenerator
  - throughput.json/csv and llm_generation_stats.json are exported at the end
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from datasets import load_dataset

from preprocessing import preprocess_documents
from embedder import Embedder, EmbedderConfig
from vectorstore import VectorStore, VectorStoreConfig
from memory import MemoryConfig, MemoryModule
from agents import RagAgent, AgentConfig, LLMConfig, LLMGenerator
from utils import get_hpc_shard

from download_wikitext_eval import download_wikitext_eval

from metrics import (
    metrics,
    summarize_throughput,
    export_throughput_json,
    export_throughput_csv,
    plot_throughput_matplotlib,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep RC RAG + Agentic Memory (HPC-friendly)")

    # Corpus input (build vector store from this)
    p.add_argument("--data_dir", type=str, default="data",
                   help="Path to input corpus directory (txt/md/json/csv/etc.)")

    p.add_argument("--max_chars", type=int, default=1000, help="Max characters per chunk.")
    p.add_argument("--overlap_chars", type=int, default=200, help="Overlap between chunks.")

    # Embeddings
    p.add_argument("--embed_model", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="HF model name for embeddings.")
    p.add_argument("--embed_batch_size", type=int, default=32, help="Embedding batch size.")
    p.add_argument("--embed_max_len", type=int, default=512, help="Max tokens for embed encoder.")

    # Vector store
    p.add_argument("--use_faiss", action="store_true", help="Use FAISS for vector search.")
    p.add_argument("--faiss_ip", action="store_true",
                   help="Use IndexFlatIP (inner product) instead of IndexFlatL2.")

    # LLM
    p.add_argument("--llm", type=str, default="gpt2", help="HF causal LM name.")
    p.add_argument("--llm_max_input", type=int, default=512, help="Max LLM input length.")
    p.add_argument("--llm_max_new_tokens", type=int, default=128, help="Max new tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")

    # Device
    p.add_argument("--device", type=str, default=None,
                   help='"cpu" or "cuda". If None, auto-detect.')

    # Evaluation mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--eval_local", action="store_true",
                      help="Run evaluation from local files in --eval_dir.")
    mode.add_argument("--eval_hf", action="store_true",
                      help="Run evaluation from Hugging Face dataset (wikitext by default).")

    # Local eval
    p.add_argument("--eval_dir", type=str, default="data/wikitext_eval",
                   help="Directory containing eval files (txt/md/log/json/jsonl).")
    p.add_argument("--eval_glob", type=str, default="*",
                   help="Glob pattern inside eval_dir (default '*').")

    # HF eval (Wikitext default)
    p.add_argument("--eval_dataset_name", type=str, default="wikitext")
    p.add_argument("--eval_subset_name", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--eval_text_column", type=str, default="text")

    # Shared eval knobs
    p.add_argument("--eval_max_samples", type=int, default=50,
                   help="Max number of eval samples per rank (local or HF).")

    # Convenience: auto-download local eval files (wikitext split dumped to txt)
    p.add_argument("--auto_download_eval", action="store_true",
                   help="Download wikitext eval set into --eval_dir before running.")

    # Output paths
    p.add_argument("--lat_csv", type=str, default="latencies.csv")
    p.add_argument("--lat_json", type=str, default="latencies.json")
    p.add_argument("--throughput_csv", type=str, default="throughput.csv")
    p.add_argument("--throughput_json", type=str, default="throughput.json")
    p.add_argument("--throughput_png", type=str, default="throughput.png")
    p.add_argument("--llm_stats_json", type=str, default="llm_generation_stats.json")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline build
# ---------------------------------------------------------------------------

def build_pipeline(args: argparse.Namespace) -> Tuple[RagAgent, int]:
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # 1) Preprocess corpus -> chunks
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Corpus directory/file not found: {args.data_dir}")

    logger.info("Preprocessing corpus from %s", args.data_dir)
    chunks, metadatas = preprocess_documents(
        input_path=args.data_dir,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )

    if not chunks:
        raise RuntimeError(
            f"No chunks generated from corpus {args.data_dir}. "
            f"Check that it contains text/JSON/CSV files."
        )

    # 2) Embed corpus
    emb_config = EmbedderConfig(
        model_name=args.embed_model,
        device=device,
        max_length=args.embed_max_len,
        batch_size=args.embed_batch_size,
    )
    embedder = Embedder(emb_config)

    logger.info("Embedding %d corpus chunks...", len(chunks))
    corpus_embeddings = embedder.embed_corpus(chunks)
    dim = int(corpus_embeddings.shape[1])
    logger.info("Corpus embeddings shape: %s", tuple(corpus_embeddings.shape))

    # 3) Vector store
    index_type = "IndexFlatIP" if args.faiss_ip else "IndexFlatL2"
    vs_config = VectorStoreConfig(
        dim=dim,
        use_faiss=args.use_faiss,
        faiss_index_type=index_type,
        normalize_embeddings=True,
    )
    vectorstore = VectorStore(vs_config)
    vectorstore.add_documents(corpus_embeddings, chunks, metadatas)

    # 4) Memory
    mem_config = MemoryConfig(dim=dim)
    memory = MemoryModule(mem_config)

    # 5) LLM
    llm_config = LLMConfig(
        model_name=args.llm,
        device=device,
        max_input_length=args.llm_max_input,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.temperature,
    )
    llm = LLMGenerator(llm_config)

    # 6) Agent
    agent = RagAgent(embedder, vectorstore, memory, llm, AgentConfig())
    return agent, dim


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _deterministic_list_files(eval_dir: str, pattern: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(eval_dir, pattern)))
    paths = [p for p in paths if os.path.isfile(p)]
    return paths


def _load_queries_from_file(path: str) -> List[Dict[str, Any]]:
    """
    Supported formats:
      - .txt/.md/.log  -> one query per file (whole content)
      - .json/.jsonl   -> dict or list entries containing 'query' or 'text'
    """
    out: List[Dict[str, Any]] = []
    suffix = Path(path).suffix.lower()

    if suffix in {".txt", ".md", ".log"}:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                out.append({"id": path, "query": text})
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc)
        return out

    if suffix in {".json", ".jsonl"}:
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            is_jsonl = (suffix == ".jsonl") or (len(lines) > 1 and all(ln.startswith("{") for ln in lines))

            if is_jsonl:
                for i, ln in enumerate(lines):
                    try:
                        rec = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    q = rec.get("query") or rec.get("text")
                    if q:
                        out.append({"id": f"{path}#line{i}", "query": str(q)})
            else:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    q = obj.get("query") or obj.get("text")
                    if q:
                        out.append({"id": path, "query": str(q)})
                elif isinstance(obj, list):
                    for i, rec in enumerate(obj):
                        if not isinstance(rec, dict):
                            continue
                        q = rec.get("query") or rec.get("text")
                        if q:
                            out.append({"id": f"{path}#idx{i}", "query": str(q)})
        except Exception as exc:
            logger.error("Failed to parse JSON from %s: %s", path, exc)

    return out


def run_eval_local_files(agent: RagAgent, args: argparse.Namespace) -> None:
    """
    Deterministic sharding by FILE across ranks (robust for Slurm).
    Each rank parses only its own files, then runs agent.generate_answer().
    """
    rank, world_size = get_hpc_shard()

    eval_dir = args.eval_dir
    if args.auto_download_eval:
        # download_wikitext_eval returns path; we honor --eval_dir as output
        eval_dir = download_wikitext_eval(output_dir=eval_dir, max_samples=args.eval_max_samples)

    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"eval_dir not found: {eval_dir}")

    files = _deterministic_list_files(eval_dir, args.eval_glob)
    if not files:
        logger.warning("No eval files in %s (glob=%s)", eval_dir, args.eval_glob)
        return

    local_files = [p for i, p in enumerate(files) if (i % world_size) == rank]
    logger.info(
        "Eval(local): rank=%d world_size=%d | local_files=%d total_files=%d | eval_dir=%s",
        rank, world_size, len(local_files), len(files), eval_dir
    )

    local_queries: List[Dict[str, Any]] = []
    for p in local_files:
        local_queries.extend(_load_queries_from_file(p))

    if not local_queries:
        logger.warning("Rank %d parsed 0 queries from %d local files.", rank, len(local_files))
        return

    # Apply per-rank cap AFTER sharding for scaling fairness
    if args.eval_max_samples is not None:
        local_queries = local_queries[: args.eval_max_samples]

    logger.info("Eval(local): rank=%d running %d queries", rank, len(local_queries))

    for qi, rec in enumerate(local_queries):
        query = str(rec["query"]).strip()
        if not query:
            continue

        answer, _debug = agent.generate_answer(query, store_in_memory=True)

        print("\n" + "=" * 80)
        print(f"[LOCAL EVAL] rank={rank} q={qi+1}/{len(local_queries)} id={rec.get('id','NA')}")
        print("- Query:")
        print(query[:500])
        print("\n- Answer:")
        print(answer.strip())
        print("=" * 80)


def run_eval_hf(agent: RagAgent, args: argparse.Namespace) -> None:
    """
    HF dataset eval. Shard by ROW index across ranks.
    """
    rank, world_size = get_hpc_shard()

    logger.info(
        "Eval(HF): loading %s / %s (split=%s)",
        args.eval_dataset_name, args.eval_subset_name, args.eval_split
    )
    ds = load_dataset(
        path=args.eval_dataset_name,
        name=args.eval_subset_name,
        split=args.eval_split,
    )

    col = args.eval_text_column
    if col not in ds.column_names:
        raise ValueError(f"Column '{col}' not found. Available: {ds.column_names}")

    total = len(ds)
    logger.info("Eval(HF): dataset rows=%d", total)

    # Shard by row index
    idxs = [i for i in range(total) if (i % world_size) == rank]

    # Per-rank cap AFTER sharding
    if args.eval_max_samples is not None:
        idxs = idxs[: args.eval_max_samples]

    logger.info("Eval(HF): rank=%d world_size=%d running %d rows", rank, world_size, len(idxs))

    for j, i in enumerate(idxs):
        query = str(ds[i][col]).strip()
        if not query:
            continue

        answer, _debug = agent.generate_answer(query, store_in_memory=True)

        print("\n" + "=" * 80)
        print(f"[HF EVAL] rank={rank} row={i} ({j+1}/{len(idxs)})")
        print("- Query:")
        print(query[:500])
        print("\n- Answer:")
        print(answer.strip())
        print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    rank, world_size = get_hpc_shard()
    logger.info("HPC shard detected: rank=%d world_size=%d", rank, world_size)
    logger.info(
        "SLURM env: PROCID=%s NTASKS=%s STEP_NUM_TASKS=%s | PMI_RANK=%s PMI_SIZE=%s | OMPI_RANK=%s OMPI_SIZE=%s",
        os.environ.get("SLURM_PROCID"),
        os.environ.get("SLURM_NTASKS"),
        os.environ.get("SLURM_STEP_NUM_TASKS"),
        os.environ.get("PMI_RANK"),
        os.environ.get("PMI_SIZE"),
        os.environ.get("OMPI_COMM_WORLD_RANK"),
        os.environ.get("OMPI_COMM_WORLD_SIZE"),
    )

    # 1) Build pipeline (corpus -> vectorstore)
    agent, dim = build_pipeline(args)
    logger.info("Pipeline ready. Embedding dimension: %d", dim)

    # 2) Evaluate
    # Default behavior: prefer local eval if eval_local set OR auto_download_eval set.
    if args.eval_hf:
        run_eval_hf(agent, args)
    else:
        # local eval by default (and for your current workflow)
        run_eval_local_files(agent, args)

    # 3) Dump latency metrics
    metrics.log_summary()
    metrics.dump_csv(args.lat_csv)
    metrics.dump_json(args.lat_json)
    logger.info("Latency metrics written: %s , %s", args.lat_csv, args.lat_json)

    # 4) Throughput exports (guard matplotlib if not installed)
    summarize_throughput(metrics)
    export_throughput_json(metrics, args.throughput_json)
    export_throughput_csv(metrics, args.throughput_csv)
    logger.info("Throughput written: %s , %s", args.throughput_json, args.throughput_csv)

    try:
        plot_throughput_matplotlib(metrics, args.throughput_png)
        logger.info("Throughput plot saved: %s", args.throughput_png)
    except ModuleNotFoundError as exc:
        logger.warning("Skipping throughput plot (missing dependency): %s", exc)

    # 5) LLM generation stats (aggregated per-rank; merge offline if needed)
    llm_stats = agent.llm.get_generation_stats()
    logger.info(
        "LLM generation stats: total_tokens=%d total_time=%.3fs num_generations=%d "
        "avg_gen_time=%.4fs tokens_per_second=%.2f",
        llm_stats["total_generated_tokens"],
        llm_stats["total_generation_time_s"],
        llm_stats["num_generations"],
        (llm_stats["avg_latency_per_generation_s"] or 0.0),
        (llm_stats["tokens_per_second"] or 0.0),
    )

    # write per-rank file to avoid clobbering in multi-rank runs
    out_path = args.llm_stats_json
    if world_size > 1:
        stem, ext = os.path.splitext(out_path)
        out_path = f"{stem}.rank{rank}{ext}"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(llm_stats, f, indent=2)
    logger.info("LLM generation stats written: %s", out_path)


if __name__ == "__main__":
    main()

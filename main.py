#!/usr/bin/env python
"""
main.py

Deep RC RAG + Agentic Memory demo.
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 09/15/2025

Instead of reading user queries from stdin, this script uses the
Hugging Face Wikitext test split as the source of input queries.

Each example's `text` field from:

    wikitext / wikitext-2-raw-v1 / test

is treated as a query to the RAG+memory agent.

unset GIT_ASKPASS

Requirements:
  pip install datasets transformers faiss-cpu  # (faiss optional)
  python -m main \
  --data_dir data/wikitext \
  --use_faiss \
  --llm gpt2 \
  --eval_dataset_name wikitext \
  --eval_subset_name wikitext-2-raw-v1 \
  --eval_split test \
  --eval_text_column text \
  --eval_max_samples 20
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Tuple

import torch
from datasets import load_dataset


from preprocessing import preprocess_documents
from embedder import Embedder, EmbedderConfig
from vectorstore import VectorStore, VectorStoreConfig
from memory import MemoryConfig, MemoryModule
from agents import RagAgent, AgentConfig, LLMConfig, LLMGenerator
from download_wikitext_eval import download_wikitext_eval
import json
import logging
from utils import get_hpc_shard


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
'''
python -m main \
  --data_dir data \
  --use_faiss \
  --llm gpt2 \
  --auto_download_eval \
  --eval_max_samples 30


   python -m main \
  --data_dir data/wikitext_eval \
  --use_faiss \
  --llm gpt2
'''
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep RC RAG using Wikitext test as queries")

    # Data / preprocessing for the corpus
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to input data directory or file for the corpus.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1000,
        help="Max characters per chunk.",
    )
    parser.add_argument(
        "--overlap_chars",
        type=int,
        default=200,
        help="Overlap characters between chunks.",
    )

    # Embedding model
    parser.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name for embeddings.",
    )
    parser.add_argument(
        "--embed_batch_size",
        type=int,
        default=32,
        help="Batch size for corpus embedding.",
    )
    parser.add_argument(
        "--embed_max_len",
        type=int,
        default=512,
        help="Max token length for embedding encoder.",
    )

    # Vector store
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Use FAISS for vector search (otherwise NumPy fallback).",
    )
    parser.add_argument(
        "--faiss_ip",
        action="store_true",
        help="Use inner product FAISS index (IndexFlatIP). If not set, use L2.",
    )

    # LLM
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt2",
        help="HuggingFace causal LM name (e.g., 'gpt2').",
    )
    parser.add_argument(
        "--llm_max_input",
        type=int,
        default=512,
        help="Max token length for LLM input.",
    )
    parser.add_argument(
        "--llm_max_new_tokens",
        type=int,
        default=128,
        help="Max number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation.",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='"cpu" or "cuda". If None, auto-detect.',
    )

    # Evaluation dataset (queries)
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name for queries (default: wikitext).",
    )
    parser.add_argument(
        "--eval_subset_name",
        type=str,
        default="wikitext-2-raw-v1",
        help="HuggingFace dataset subset/config for queries.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help="Dataset split to use as query source (default: test).",
    )
    parser.add_argument(
        "--eval_text_column",
        type=str,
        default="text",
        help="Column name in the eval dataset containing query text.",
    )
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=50,
        help="Max number of eval queries to run from the dataset.",
    )

    parser.add_argument(
        "--auto_download_eval",
        action="store_true",
        help="Automatically download Wikitext evaluation dataset before running."
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_pipeline(args: argparse.Namespace) -> Tuple[RagAgent, int]:
    """
    Build the full RAG + memory pipeline:
      - preprocess corpus
      - embed corpus
      - build vector store
      - build memory
      - build LLM + agent

    Returns
    -------
    agent : RagAgent
    dim   : int  (embedding dimension)
    """
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # 1. Preprocess corpus
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

    # 2. Embed corpus
    emb_config = EmbedderConfig(
        model_name=args.embed_model,
        device=device,
        max_length=args.embed_max_len,
        batch_size=args.embed_batch_size,
    )
    embedder = Embedder(emb_config)

    logger.info("Embedding %d corpus chunks...", len(chunks))
    corpus_embeddings = embedder.embed_corpus(chunks)
    dim = corpus_embeddings.shape[1]
    logger.info("Corpus embeddings shape: %s", corpus_embeddings.shape)

    # 3. Vector store
    index_type = "IndexFlatIP" if args.faiss_ip else "IndexFlatL2"
    vs_config = VectorStoreConfig(
        dim=dim,
        use_faiss=args.use_faiss,
        faiss_index_type=index_type,
        normalize_embeddings=True,
    )
    vectorstore = VectorStore(vs_config)
    vectorstore.add_documents(corpus_embeddings, chunks, metadatas)

    # 4. Memory
    mem_config = MemoryConfig(dim=dim)
    memory = MemoryModule(mem_config)

    # 5. LLM
    llm_config = LLMConfig(
        model_name=args.llm,
        device=device,
        max_input_length=args.llm_max_input,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.temperature,
    )
    llm = LLMGenerator(llm_config)

    # 6. Agent
    agent_config = AgentConfig()
    agent = RagAgent(embedder, vectorstore, memory, llm, agent_config)

    return agent, dim


def run_wikitext_eval(agent: RagAgent, args: argparse.Namespace) -> None:
    """
    Use the Wikitext test split as the source of input queries.

    Each row's `eval_text_column` content is passed to `agent.generate_answer`.
    """
    logger.info(
        "Loading eval dataset: %s / %s (split=%s)",
        args.eval_dataset_name,
        args.eval_subset_name,
        args.eval_split,
    )
    ds = load_dataset(
        path=args.eval_dataset_name,
        name=args.eval_subset_name,
        split=args.eval_split,
    )

    if args.eval_text_column not in ds.column_names:
        raise ValueError(
            f"Column '{args.eval_text_column}' not found in dataset columns: "
            f"{ds.column_names}"
        )

    num_rows = len(ds)
    max_samples = min(args.eval_max_samples, num_rows)
    logger.info(
        "Eval dataset size: %d rows; running on first %d samples.",
        num_rows,
        max_samples,
    )

    for i in range(max_samples):
        row = ds[i]
        query = str(row[args.eval_text_column]).strip()
        if not query:
            continue

        logger.info("=== [Eval %d/%d] Query from Wikitext ===", i + 1, max_samples)
        logger.info("Query snippet: %s", query[:200].replace("\n", " "))

        answer, debug = agent.generate_answer(query, store_in_memory=True)

        # Print compactly; you can also log to file
        print("\n" + "=" * 80)
        print(f"[WIKITEXT TEST] Example {i + 1}/{max_samples}")
        print("- Query:")
        print(query[:1000])  # avoid dumping extremely long lines
        print("\n- Answer:")
        print(answer)
        print("=" * 80 + "\n")

def load_local_eval_dataset(eval_dir):
    """
    Load evaluation dataset from local .txt files created by download_wikitext_eval.
    Returns a list of strings, each used as an input query to the agent.
    """
    from pathlib import Path

    p = Path(eval_dir)
    print(p)
    files = sorted(p.glob("*.txt"))
    queries = []

    for f in files:
        q = f.read_text(encoding="utf-8").strip()
        if q:
            queries.append(q)

    return queries

'''
def run_eval_from_local_files(agent, eval_dir, max_samples):
    queries = load_local_eval_dataset(eval_dir)
    queries = queries[:max_samples]

    for i, query in enumerate(queries):
        print("\n" + "="*80)
        print(f"[EVAL] Query {i+1}/{len(queries)}")
        print(query[:500])
        print("-"*80)

        answer, debug = agent.generate_answer(query)
        print("[Answer]:\n", answer)
'''

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from utils import get_hpc_shard
from agents import RagAgent
import logging

logger = logging.getLogger(__name__)


def _load_eval_queries_from_file(path: str) -> List[Dict[str, Any]]:
    """
    Load queries from a single eval file.

    Supported formats:
      - .txt / .md / .log:
          entire file content becomes one query: {"id": path, "query": text}
      - .json / .jsonl:
          * dict with "query" or "text" field
          * list of dicts with "query" or "text" field
    """
    queries: List[Dict[str, Any]] = []
    suffix = os.path.splitext(path)[1].lower()

    # Plain text → one query per file
    if suffix in {".txt", ".md", ".log"}:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                queries.append({"id": path, "query": text})
        except Exception as exc:
            logger.error("Failed to read text file %s: %s", path, exc)

    # JSON / JSONL
    elif suffix in {".json", ".jsonl"}:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            # Try JSONL
            if suffix == ".jsonl" or "\n" in raw:
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                is_jsonl = all(ln.startswith("{") and ln.endswith("}") for ln in lines)
            else:
                is_jsonl = False

            if is_jsonl:
                for i, ln in enumerate(lines):
                    try:
                        rec = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    q = rec.get("query") or rec.get("text")
                    if q:
                        queries.append({"id": f"{path}#line{i}", "query": str(q)})
            else:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    q = obj.get("query") or obj.get("text")
                    if q:
                        queries.append({"id": path, "query": str(q)})
                elif isinstance(obj, list):
                    for i, rec in enumerate(obj):
                        if not isinstance(rec, dict):
                            continue
                        q = rec.get("query") or rec.get("text")
                        if q:
                            queries.append({"id": f"{path}#idx{i}", "query": str(q)})
        except Exception as exc:
            logger.error("Failed to read JSON from %s: %s", path, exc)

    else:
        # Unknown extension → ignore
        logger.debug("Skipping unsupported eval file type: %s", path)

    return queries


def run_eval_from_local_files(
    agent: RagAgent,
    eval_dir: str,
    max_samples: Optional[int] = None,
) -> None:
    """
    Rank-aware eval loop that reads queries from ALL local files in eval_dir.

    Supported:
      - *.txt / *.md / *.log     -> 1 query per file
      - *.json / *.jsonl         -> dict or list with 'query' or 'text' key

    Each rank (from get_hpc_shard) evaluates a disjoint subset of queries.
    """
    rank, world_size = get_hpc_shard()

    pattern = os.path.join(eval_dir, "*")
    paths = sorted(glob.glob(pattern))

    all_queries: List[Dict[str, Any]] = []
    for p in paths:
        file_queries = _load_eval_queries_from_file(p)
        all_queries.extend(file_queries)

    if not all_queries:
        logger.warning("No eval queries found under %s", eval_dir)
        return

    if max_samples is not None:
        all_queries = all_queries[:max_samples]

    # Shard queries across ranks
    local_queries = [
        (i, q) for i, q in enumerate(all_queries)
        if i % max(world_size, 1) == rank
    ]

    logger.info(
        "Rank %d/%d evaluating %d of %d queries from %s",
        rank, world_size, len(local_queries), len(all_queries), eval_dir,
    )

    # Simple stdout print; you can also log to file if you want
    for idx, rec in local_queries:
        qid = rec.get("id", f"q{idx}")
        query = rec["query"]

        answer, debug = agent.generate_answer(query)

        print("=" * 80)
        print(f"[Rank {rank}] Query #{idx} (id={qid}):")
        print(query[:500])
        print("\n[Answer]:")
        print(answer.strip())
        print("=" * 80)

def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # 0) Automatically download Wikitext test split if requested
    if args.auto_download_eval:
        eval_dir = download_wikitext_eval(
            output_dir="data/wikitext_eval",
            max_samples=args.eval_max_samples
        )
    else:
        eval_dir = args.eval_dir if hasattr(args, "eval_dir") else "data/wikitext_eval"

    # 1) Build pipeline
    agent, dim = build_pipeline(args)
    logger.info("Pipeline ready. Embedding dimension: %d", dim)

    # 2) Evaluate using local files
    run_eval_from_local_files(agent, eval_dir, args.eval_max_samples)

    # 3) Dump metrics
    metrics.log_summary()
    metrics.dump_csv("latencies.csv")
    metrics.dump_json("latencies.json")
    logger.info("Latency metrics written.")


    # ...after entire pipeline finishes:
    summarize_throughput(metrics)
    export_throughput_json(metrics, "throughput.json")
    export_throughput_csv(metrics, "throughput.csv")
    # If you want an image saved instead of on-screen:
    plot_throughput_matplotlib(metrics, "throughput.png")


    # Sanity check: force one LLM call so we know stats work
    #test_query = "RADICAL-Pilot provides pilot-job abstraction and deterministic scheduling for HPC workflows."
    #answer, debug = agent.generate_answer(test_query)
    #logger.info("Test query answer (truncated): %s", answer[:200])

    
    # ⬇️ NEW: LLM generation token stats
    llm_stats = agent.llm.get_generation_stats()

    logger.info(
        "LLM generation stats: total_tokens=%d, total_time=%.3fs, "
        "num_generations=%d, avg_gen_time=%.4fs, tokens_per_second=%.2f",
        llm_stats["total_generated_tokens"],
        llm_stats["total_generation_time_s"],
        llm_stats["num_generations"],
        (llm_stats["avg_latency_per_generation_s"] or 0.0),
        (llm_stats["tokens_per_second"] or 0.0),
    )

    # Also dump to a JSON file for paper/post-processing
    with open("llm_generation_stats.json", "w", encoding="utf-8") as f:
        json.dump(llm_stats, f, indent=2)
    logger.info("LLM generation stats written to llm_generation_stats.json")




if __name__ == "__main__":
    main()

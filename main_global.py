# main_global.py
'''
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 15/12/2025
Descriptions: Parallelism-2, Rank 0 is responsible tokenizing and saving to vector store; other rank is responsible to load and use it! 
'''
from __future__ import annotations

import argparse
import json
import logging
import os

import torch

from embedder import Embedder, EmbedderConfig
from memory import MemoryConfig, MemoryModule
from agents import RagAgent, AgentConfig, LLMConfig, LLMGenerator
from global_index import build_or_load_global_vectorstore
from utils import get_hpc_shard
from metrics import metrics

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser("Deep RC RAG - GLOBAL VectorStore Mode")

    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--global_index_dir", type=str, required=True)

    p.add_argument("--max_chars", type=int, default=1000)
    p.add_argument("--overlap_chars", type=int, default=200)
    p.add_argument("--file_glob", type=str, default="*")

    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embed_batch_size", type=int, default=32)
    p.add_argument("--embed_max_len", type=int, default=512)

    p.add_argument("--use_faiss", action="store_true")
    p.add_argument("--faiss_ip", action="store_true")

    p.add_argument("--llm", type=str, default="gpt2")
    p.add_argument("--llm_max_input", type=int, default=512)
    p.add_argument("--llm_max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)

    p.add_argument("--device", type=str, default=None)

    # Eval options (reuse local eval from main.py if present)
    p.add_argument("--eval_dir", type=str, default=None)
    p.add_argument("--eval_glob", type=str, default="*")
    p.add_argument("--eval_max_samples", type=int, default=50)

    p.add_argument("--lat_csv", type=str, default="latencies.global.csv")
    p.add_argument("--lat_json", type=str, default="latencies.global.json")

    p.add_argument("--llm_stats_json", type=str, default="llm_generation_stats.global.json")
    p.add_argument("--auto_download_eval", action="store_true", help="Automatically download evaluation dataset if not present.",)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    rank, world_size = get_hpc_shard()
    logger.info("GLOBAL MODE: rank=%d world_size=%d", rank, world_size)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)

    # Embedder (used by rank0 to build, and by all ranks for queries)
    embedder = Embedder(
        EmbedderConfig(
            model_name=args.embed_model,
            device=device,
            max_length=args.embed_max_len,
            batch_size=args.embed_batch_size,
        )
    )

    # Determine dim cheaply
    dim = int(embedder.embed_query("dim_probe").shape[0])
    logger.info("Embedding dim=%d", dim)

    faiss_index_type = "IndexFlatIP" if args.faiss_ip else "IndexFlatL2"

    vectorstore = build_or_load_global_vectorstore(
        data_dir=args.data_dir,
        embedder=embedder,
        dim=dim,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        file_glob=args.file_glob,
        global_index_dir=args.global_index_dir,
        faiss_index_type=faiss_index_type,
        normalize_embeddings=True,
    )

    memory = MemoryModule(MemoryConfig(dim=dim))
    llm = LLMGenerator(
        LLMConfig(
            model_name=args.llm,
            device=device,
            max_input_length=args.llm_max_input,
            max_new_tokens=args.llm_max_new_tokens,
            temperature=args.temperature,
        )
    )

    agent = RagAgent(embedder, vectorstore, memory, llm, AgentConfig())

    if args.eval_dir:
        # Import your eval function from baseline main.py for code reuse.
        from main import run_eval_local_files  # type: ignore
        run_eval_local_files(agent, args)
    else:
        q = "RADICAL-Pilot provides pilot-job abstraction and deterministic scheduling for HPC workflows."
        ans, _ = agent.generate_answer(q)
        if rank == 0:
            print(ans)

    # Dump per-rank latencies
    metrics.dump_csv(args.lat_csv.replace(".csv", f".rank{rank}.csv") if world_size > 1 else args.lat_csv)
    metrics.dump_json(args.lat_json.replace(".json", f".rank{rank}.json") if world_size > 1 else args.lat_json)

    # LLM stats
    stats = agent.llm.get_generation_stats()
    out = args.llm_stats_json
    if world_size > 1:
        stem, ext = os.path.splitext(out)
        out = f"{stem}.rank{rank}{ext}"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("GLOBAL MODE done.")


if __name__ == "__main__":
    main()

'''
export IDX=/sfs/gpfs/tardis/project/bi_dsc_community/drc_rag/shared_index_wikitext
'''
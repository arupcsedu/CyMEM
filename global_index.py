# global_index.py
'''
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 13/12/2025
Descriptions: Parallelism-2, Rank 0 is responsible tokenizing and saving to vector store; other rank is responsible to load and use it! 
'''

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from preprocessing import preprocess_documents
from embedder import Embedder
from vectorstore import VectorStore, VectorStoreConfig
from utils import get_hpc_shard

from metrics import record_latency

logger = logging.getLogger(__name__)



def wait_for_global_index(index_dir: Path, timeout_s: int = 600):
    """
    Block until global index is ready (created by rank 0).
    """
    ready_flag = index_dir / "READY"
    start = time.time()

    while not ready_flag.exists():
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for global index at {index_dir}"
            )
        time.sleep(0.5)

    logger.info("Detected READY flag at %s", ready_flag)



def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def _wait_for_file(path: Path, timeout_s: int = 3600, poll_s: float = 1.0) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for sentinel: {path}")
        time.sleep(poll_s)


def _save_docs_jsonl(vs: VectorStore, docs_path: Path) -> None:
    # Your VectorStore uses private fields: _texts, _metadatas
    texts = getattr(vs, "_texts", None)
    metas = getattr(vs, "_metadatas", None)
    if texts is None or metas is None:
        raise RuntimeError("VectorStore missing _texts/_metadatas; cannot persist docs.")

    if len(texts) != len(metas):
        raise RuntimeError(f"text/metadata length mismatch: {len(texts)} vs {len(metas)}")

    records = [{"text": t, "metadata": m} for t, m in zip(texts, metas)]
    _write_jsonl(docs_path, records)


def _load_docs_jsonl(vs: VectorStore, docs_path: Path) -> None:
    docs = _read_jsonl(docs_path)
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]

    # restore into your VectorStore internals
    vs._texts = texts          # type: ignore[attr-defined]
    vs._metadatas = metas      # type: ignore[attr-defined]


def _save_docs(vs, path: Path):
    import json
    with open(path, "w") as f:
        for text, meta in zip(vs._texts, vs._metadatas):
            f.write(json.dumps({"text": text, "metadata": meta}) + "\n")


def _load_docs(vs, path: Path):
    import json
    texts, metas = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append(rec["metadata"])
    vs._texts = texts
    vs._metadatas = metas



def build_or_load_global_vectorstore(
    *,
    data_dir: str | None = None,
    input_path: str | None = None,
    embedder,
    dim: int,
    global_index_dir: str,
    max_chars: int = 1000,
    overlap_chars: int = 200,
    file_glob: str = "*",
    faiss_index_type: str = "IndexFlatL2",
    normalize_embeddings: bool = True,
    timeout_s: int = 3600,
):
    """
    Build a global FAISS vectorstore on rank 0 and load it on other ranks.
    """

    # -----------------------------
    # Resolve input path
    # -----------------------------
    if input_path is None and data_dir is None:
        raise ValueError("Provide either data_dir=... or input_path=...")
    if input_path is None:
        input_path = data_dir

    # -----------------------------
    # HPC rank info
    # -----------------------------
    rank, world_size = get_hpc_shard()

    # -----------------------------
    # Paths
    # -----------------------------
    index_dir = Path(global_index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "index.faiss"
    docs_path = index_dir / "docs.jsonl"
    ready_flag = index_dir / "READY"

    # -----------------------------
    # VectorStore config
    # -----------------------------
    vs_cfg = VectorStoreConfig(
        dim=dim,
        use_faiss=True,
        faiss_index_type=faiss_index_type,
        normalize_embeddings=normalize_embeddings,
    )
    vectorstore = VectorStore(vs_cfg)

    # -----------------------------
    # Rank 0 builds index
    # -----------------------------
    # Rank 0 builds FULL index
    if rank == 0:
        logger.info("Rank 0 building GLOBAL vectorstore from %s", input_path)
        os.environ["SLURM_PROCID"] = "0"
        os.environ["SLURM_NTASKS"] = "1"

        chunks, metadatas = preprocess_documents(
            input_path=input_path,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            file_glob=file_glob
        )

        embeddings = embedder.embed_corpus(chunks)

        vectorstore.add_documents(
            embeddings=embeddings,
            texts=chunks,
            metadatas=metadatas,
        )

        vectorstore.save(index_path)
        _save_docs(vectorstore, docs_path)
        ready_flag.touch()


    # -----------------------------
    # Other ranks wait + load
    # -----------------------------
    else:
        logger.info("Rank %d waiting for global index...", rank)
        wait_for_global_index(index_dir, timeout_s=timeout_s)

        logger.info("Rank %d loading global vectorstore", rank)
        vectorstore.load(index_path)
        _load_docs(vectorstore, docs_path)

    return vectorstore



def load_global_vectorstore(
    *,
    dim: int,
    index_path: Path,
    docs_path: Path,
    faiss_index_type: str = "IndexFlatL2",
    normalize_embeddings: bool = True,
) -> VectorStore:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing docs file: {docs_path}")

    cfg = VectorStoreConfig(
        dim=dim,
        use_faiss=True,
        faiss_index_type=faiss_index_type,
        normalize_embeddings=normalize_embeddings,
    )
    vs = VectorStore(cfg)

    # Load FAISS index via your built-in method
    vs.load(str(index_path))

    # Load docs into internal arrays
    _load_docs_jsonl(vs, docs_path)

    logger.info("Loaded GLOBAL vectorstore with %d docs", len(vs._texts))  # type: ignore[attr-defined]
    return vs

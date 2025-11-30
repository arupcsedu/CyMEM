"""
preprocessing.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 10/26/2025

Document loading, cleaning, and chunking utilities for Deep RC RAG.

This module is responsible for turning raw files (txt, md, json, csv, etc.)
into a list of smaller text chunks plus associated metadata that can be
fed into the embedder and vector store.

In addition, we record latency metrics for:
  - overall run (`preprocessing_total_run`)                 -> 1 per main() run
  - per-document pipeline (`preprocessing_documents`)       -> 1 per raw document
  - load stage (`preprocess.load`)                          -> 1 per file
  - cleaning stage (`preprocess.clean`)                     -> 1 per raw document
  - chunking stage per document (`preprocess.chunk`)        -> 1 per cleaned document
  - chunking stage per chunk (`preprocess.chunk_per_chunk`) -> 1 per chunk

The metrics are collected via the global recorder in `metrics.py`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from utils import get_hpc_shard

try:
    import pandas as pd  # optional, used only for CSV/TSV
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

from metrics import record_latency


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Represents a single text chunk derived from a source document."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, str]


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    # Collapse all whitespace runs into a single space
    text = _WHITESPACE_RE.sub(" ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Simple character-level chunking with overlap.

    Parameters
    ----------
    text : str
        Input text to chunk.
    max_chars : int
        Maximum size of a chunk.
    overlap_chars : int
        Number of characters to overlap between consecutive chunks.

    Returns
    -------
    List[str]
        List of chunk strings.
    """
    if not text:
        return []

    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative.")
    if overlap_chars >= max_chars:
        logger.warning(
            "overlap_chars (%d) >= max_chars (%d); resetting overlap to 0.",
            overlap_chars,
            max_chars,
        )
        overlap_chars = 0

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap_chars

    return chunks


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read %s: %s", path, exc)
        return ""


def _read_json_file(path: Path, text_key: str = "text") -> List[Tuple[str, Dict[str, str]]]:
    """
    Read JSON or JSONL where each entry contains a `text_key` field.
    Returns list of (text, metadata) pairs.
    """
    out: List[Tuple[str, Dict[str, str]]] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        # Try JSONL first
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) > 1 and all(ln.startswith("{") for ln in lines):
            for i, ln in enumerate(lines):
                rec = json.loads(ln)
                text = str(rec.get(text_key, ""))
                meta = {k: str(v) for k, v in rec.items() if k != text_key}
                out.append((text, meta | {"source_file": str(path), "line": str(i)}))
        else:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                text = str(obj.get(text_key, ""))
                meta = {k: str(v) for k, v in obj.items() if k != text_key}
                out.append((text, meta | {"source_file": str(path)}))
            elif isinstance(obj, list):
                for i, rec in enumerate(obj):
                    if not isinstance(rec, dict):
                        continue
                    text = str(rec.get(text_key, ""))
                    meta = {k: str(v) for k, v in rec.items() if k != text_key}
                    out.append((text, meta | {"source_file": str(path), "index": str(i)}))
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read JSON from %s: %s", path, exc)
    return out


def _read_table_file(path: Path, text_column: str = "text") -> List[Tuple[str, Dict[str, str]]]:
    """
    Read CSV/TSV using pandas (if available). The `text_column` is treated
    as the main text; all other columns become metadata.
    """
    if pd is None:
        logger.warning("pandas is not available; skipping table file %s", path)
        return []

    sep = "\t" if path.suffix.lower() in {".tsv"} else ","
    out: List[Tuple[str, Dict[str, str]]] = []
    try:
        df = pd.read_csv(path, sep=sep)
        if text_column not in df.columns:
            logger.warning(
                "Column '%s' not found in %s; using first column as text.",
                text_column,
                path,
            )
            text_col = df.columns[0]
        else:
            text_col = text_column

        for idx, row in df.iterrows():
            text = str(row[text_col])
            meta = {str(c): str(row[c]) for c in df.columns if c != text_col}
            meta["source_file"] = str(path)
            meta["row"] = str(idx)
            out.append((text, meta))
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read table from %s: %s", path, exc)
    return out


def load_raw_documents(
    input_path: str | Path,
    text_key: str = "text",
    table_text_column: str = "text",
    file_glob: str = "*",
    rank: int = 0,
    world_size: int = 1,
) -> List[Tuple[str, Dict[str, str]]]:
    """
    Load raw documents from a directory or single file.

    If world_size > 1, shard the file list across ranks using
    a simple modulo scheme: file_idx % world_size == rank.
     - `preprocess.load`: 1 sample per file (not per document pair).
    """
    path = Path(input_path)
    pairs: List[Tuple[str, Dict[str, str]]] = []

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        all_files = sorted(path.glob(file_glob))
    else:
        all_files = [path]

    # Shard files across ranks
    files = [
        f for i, f in enumerate(all_files)
        if i % max(world_size, 1) == rank and not f.is_dir()
    ]

    logger.info(
        "Rank %d/%d loading %d of %d files from %s",
        rank, world_size, len(files), len(all_files), input_path,
    )

    for f in files:
        # Time each file load separately:
        with record_latency("preprocess.load", store_samples=True):
            suffix = f.suffix.lower()
            if suffix in {".txt", ".md", ".log"}:
                text = _read_text_file(f)
                if text:
                    pairs.append((text, {"source_file": str(f)}))

            elif suffix in {".json", ".jsonl"}:
                pairs.extend(_read_json_file(f, text_key=text_key))

            elif suffix in {".csv", ".tsv"}:
                pairs.extend(_read_table_file(f, text_column=table_text_column))

            else:
                text = _read_text_file(f)
                if text:
                    pairs.append((text, {"source_file": str(f)}))

    logger.info(
        "Rank %d/%d loaded %d raw document(s) from %s",
        rank, world_size, len(pairs), input_path,
    )
    return pairs
'''
def load_raw_documents(
    input_path: str | Path,
    text_key: str = "text",
    table_text_column: str = "text",
    file_glob: str = "*",
) -> List[Tuple[str, Dict[str, str]]]:
    """
    Load raw documents from a directory or single file.

    Returns
    -------
    List[Tuple[str, Dict[str, str]]]
        A list of (text, metadata) pairs. Each pair is considered a
        "raw document" for the purposes of cleaning and chunking.

    Metrics
    -------
    - `preprocess.load`: 1 sample per file (not per document pair).
    """
    path = Path(input_path)
    pairs: List[Tuple[str, Dict[str, str]]] = []

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        files = sorted(path.glob(file_glob))
    else:
        files = [path]

    for f in files:
        # skip directories
        if f.is_dir():
            continue

        # Time each file load separately:
        #  → "preprocess.load" count == number of files processed
        with record_latency("preprocess.load", store_samples=True):
            suffix = f.suffix.lower()
            if suffix in {".txt", ".md", ".log"}:
                text = _read_text_file(f)
                if text:
                    pairs.append((text, {"source_file": str(f)}))

            elif suffix in {".json", ".jsonl"}:
                pairs.extend(_read_json_file(f, text_key=text_key))

            elif suffix in {".csv", ".tsv"}:
                pairs.extend(_read_table_file(f, text_column=table_text_column))

            else:
                # treat unknown extensions as text
                text = _read_text_file(f)
                if text:
                    pairs.append((text, {"source_file": str(f)}))

    logger.info("Loaded %d raw document(s) from %s", len(pairs), input_path)
    return pairs
'''

# ---------------------------------------------------------------------------
# High-level preprocessing entry point
# ---------------------------------------------------------------------------

def preprocess_documents(
    input_path: str | Path,
    max_chars: int = 1000,
    overlap_chars: int = 200,
    text_key: str = "text",
    table_text_column: str = "text",
    file_glob: str = "*",
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    High-level preprocessing function used by the rest of the RAG pipeline.

    Steps (per document):
      1. Clean text.
      2. Split into overlapping chunks.

    Metrics
    -------
    - `preprocessing_total_run`: 1 sample for the entire corpus preprocessing.
    - `preprocessing_documents`: 1 sample per raw document (clean+chunk).
    - `preprocess.clean`: 1 sample per raw document.
    - `preprocess.chunk`: 1 sample per cleaned document (chunk_text call).
    - `preprocess.chunk_per_chunk`: 1 sample per final chunk.

    Returns
    -------
    chunks : List[str]
        List of text chunks.
    metadatas : List[Dict[str, str]]
        List of metadata dicts, aligned 1:1 with `chunks`.
    """
    with record_latency("preprocessing_total_run", store_samples=True):

        # 1) Load all raw documents (per-file timing handled inside)
        raw_pairs = load_raw_documents(
            input_path=input_path,
            text_key=text_key,
            table_text_column=table_text_column,
            file_glob=file_glob,
        )

        num_raw = len(raw_pairs)
        num_cleaned = 0

        chunks: List[str] = []
        metadatas: List[Dict[str, str]] = []

        # 2) Per-document pipeline: clean + chunk
        for doc_idx, (text, meta) in enumerate(raw_pairs):
            # Full per-document pipeline timing
            with record_latency("preprocessing_documents", store_samples=True):

                # Clean (per document)
                with record_latency("preprocess.clean", store_samples=True):
                    cleaned = clean_text(text)

                if not cleaned:
                    continue

                num_cleaned += 1
                doc_id = meta.get("doc_id", f"doc-{doc_idx}")

                # Chunking per document (chunk_text call)
                with record_latency("preprocess.chunk", store_samples=True):
                    doc_chunks = chunk_text(
                        cleaned,
                        max_chars=max_chars,
                        overlap_chars=overlap_chars,
                    )

                # Per-chunk timing for metadata creation / append
                for chunk_idx, chunk in enumerate(doc_chunks):
                    with record_latency("preprocess.chunk_per_chunk", store_samples=True):
                        chunk_id = f"{doc_id}-chunk-{chunk_idx}"
                        chunk_meta = dict(meta)  # shallow copy
                        chunk_meta["doc_id"] = doc_id
                        chunk_meta["chunk_id"] = chunk_id
                        chunks.append(chunk)
                        metadatas.append(chunk_meta)

        logger.info(
            "Cleaned %d document(s); %d discarded as empty.",
            num_raw,
            num_raw - num_cleaned,
        )
        logger.info(
            "Preprocessing complete: produced %d chunk(s) from %d cleaned document(s).",
            len(chunks),
            num_cleaned,
        )

    return chunks, metadatas

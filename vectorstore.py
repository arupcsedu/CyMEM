"""
vectorstore.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 22/11/2025
Descriptions: Vector store abstraction for Deep RC RAG.


This module provides a simple interface around FAISS (if available) or a
NumPy-based fallback for similarity search. It stores dense embeddings for
document chunks along with their texts and metadata, and supports:

  - add_documents(embeddings, texts, metadatas)
  - search(query_embedding, top_k)

In addition, we record latency metrics for:
  - context_store_vectorstore
  - context_load_vectorstore

via the global metrics recorder in `metrics.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from metrics import record_latency


logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - FAISS is optional
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VectorStoreConfig:
    dim: int                          # dimensionality of the embeddings
    use_faiss: bool = True            # whether to use FAISS if available
    faiss_index_type: str = "IndexFlatIP"  # or "IndexFlatL2"
    normalize_embeddings: bool = True # if using inner product, ensure normalized
    index_path: Optional[str] = None  # optional path to save/load FAISS index


# ---------------------------------------------------------------------------
# Main vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """Vector store for document chunks, with optional FAISS acceleration."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config

        if config.use_faiss and not _FAISS_AVAILABLE:
            logger.warning("FAISS requested but not available; using NumPy fallback.")
            self.config.use_faiss = False

        self._dim = config.dim

        # Backing index
        if self.config.use_faiss:
            self._index = self._build_faiss_index()
            logger.info("VectorStore initialized with FAISS (%s).", config.faiss_index_type)
        else:
            self._index = None  # NumPy fallback: we store everything in _embeddings

        # Storage for texts and metadata, aligned with embeddings
        self._embeddings: Optional[np.ndarray] = None  # (N, dim) when not using FAISS
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return len(self._texts)

    def add_documents(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add a batch of document chunks to the vector store.

        Parameters
        ----------
        embeddings : np.ndarray
            Array of shape (N, dim).
        texts : List[str]
            List of N text chunks.
        metadatas : List[Dict[str, Any]]
            List of N metadata dicts.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Expected embeddings shape (N, {self._dim}), "
                f"got {embeddings.shape}."
            )
        if len(texts) != embeddings.shape[0] or len(metadatas) != embeddings.shape[0]:
            raise ValueError(
                "Embeddings, texts, and metadatas must have the same length."
            )

        N = embeddings.shape[0]
        if N == 0:
            return

        with record_latency("context_store_vectorstore", store_samples=True):
            if self.config.normalize_embeddings:
                embeddings = _l2_normalize(embeddings)

            if self.config.use_faiss:
                # Add to FAISS index
                emb_f32 = embeddings.astype("float32")
                self._index.add(emb_f32)
            else:
                # Append to NumPy backing array
                if self._embeddings is None:
                    self._embeddings = embeddings.astype("float32")
                else:
                    self._embeddings = np.vstack(
                        [self._embeddings, embeddings.astype("float32")]
                    )

            # Append texts and metadata
            self._texts.extend(texts)
            self._metadatas.extend(metadatas)

        logger.info(
            "Added %d documents to vector store. New size=%d",
            N,
            self.size,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for a single query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            1D array of shape (dim,) or 2D array of shape (1, dim).
        top_k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        List[Dict[str, Any]]
            List of results, each containing fields:
              - "score": float
              - "text": str
              - "metadata": Dict[str, Any]
        """
        if self.size == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]
        if query_embedding.shape[1] != self._dim:
            raise ValueError(
                f"Expected query embedding shape (*, {self._dim}), "
                f"got {query_embedding.shape}."
            )

        with record_latency("context_load_vectorstore", store_samples=True):
            if self.config.normalize_embeddings:
                q = _l2_normalize(query_embedding.astype("float32"))
            else:
                q = query_embedding.astype("float32")

            if self.config.use_faiss:
                scores, indices = self._faiss_search(q, top_k)
            else:
                scores, indices = self._numpy_search(q, top_k)

        # Build result objects
        results: List[Dict[str, Any]] = []
        # q is shape (1, dim), so we use scores[0] and indices[0]
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= self.size:
                continue
            results.append(
                {
                    "score": float(score),
                    "text": self._texts[idx],
                    "metadata": self._metadatas[idx],
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # (Optional) persistence helpers for FAISS index
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None) -> None:
        """
        Save the FAISS index (if used) to disk. The texts and metadata
        are not serialized here (that should be done separately if needed).
        """
        if not self.config.use_faiss or self._index is None:
            logger.info("VectorStore.save(): no FAISS index to save.")
            return

        index_path = path or self.config.index_path
        if not index_path:
            logger.warning("VectorStore.save(): no index_path given; skipping.")
            return

        logger.info("Saving FAISS index to %s", index_path)
        faiss.write_index(self._index, index_path)

    def load(self, path: Optional[str] = None) -> None:
        """
        Load a FAISS index from disk. This assumes the texts and metadata
        are restored separately and kept consistent with the index ordering.
        """
        if not self.config.use_faiss:
            logger.warning("VectorStore.load(): use_faiss=False; nothing to load.")
            return

        index_path = path or self.config.index_path
        if not index_path:
            logger.warning("VectorStore.load(): no index_path given; skipping.")
            return

        logger.info("Loading FAISS index from %s", index_path)
        self._index = faiss.read_index(index_path)

    # ------------------------------------------------------------------ #
    # Internal FAISS / NumPy search
    # ------------------------------------------------------------------ #

    def _build_faiss_index(self):
        if self.config.faiss_index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(self._dim)
        elif self.config.faiss_index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(self._dim)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.config.faiss_index_type}")
        return index

    def _faiss_search(
        self,
        query: np.ndarray,   # shape (1, dim)
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS-based similarity search."""
        k = min(top_k, self.size)
        scores, indices = self._index.search(query, k)  # type: ignore[attr-defined]
        return scores, indices

    def _numpy_search(
        self,
        query: np.ndarray,  # shape (1, dim)
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy-based similarity search (fallback when FAISS is not available)."""
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            # No data
            scores = np.zeros((1, 0), dtype=np.float32)
            indices = np.zeros((1, 0), dtype=np.int64)
            return scores, indices

        # Query and stored embeddings should already be normalized if requested
        emb = self._embeddings  # (N, dim)
        q = query[0]            # (dim,)

        if self.config.faiss_index_type == "IndexFlatIP":
            # Use inner product (cosine if normalized)
            sims = emb @ q  # (N,)
        else:
            # Use negative L2 distance so that larger is better
            diffs = emb - q[None, :]
            dists = np.sum(diffs * diffs, axis=1)
            sims = -dists  # larger is better

        k = min(top_k, emb.shape[0])
        if k <= 0:
            scores = np.zeros((1, 0), dtype=np.float32)
            indices = np.zeros((1, 0), dtype=np.int64)
            return scores, indices

        top_idx = np.argpartition(-sims, k - 1)[:k]
        # Sort within top-k
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        scores = sims[top_idx].astype("float32")[None, :]
        indices = top_idx.astype("int64")[None, :]
        return scores, indices


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors row-wise.

    Parameters
    ----------
    x : np.ndarray
        Shape (N, D) or (D,).

    Returns
    -------
    np.ndarray
        Same shape as input, normalized.
    """
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms

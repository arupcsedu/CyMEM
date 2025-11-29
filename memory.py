"""
memory.py

Hierarchical memory module for Agentic RAG in Deep RC RAG.

This module provides:
  - Short-Term Memory (STM): recent interactions (text-level)
  - Long-Term Memory (LTM): vector-based semantic memory of important facts
  - Episodic Memory (EM): compressed summaries of episodes/sessions

It exposes a single high-level API:

  - load_context(query_embedding, ...) -> dict
  - store_interaction(...)

and records latency metrics for:
  - context_load_memory
  - context_load_memory.stm
  - context_load_memory.ltm
  - context_load_memory.em
  - context_store_memory

via the global recorder in `metrics.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from metrics import record_latency


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config and data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    """Configuration for the hierarchical memory module."""
    dim: int                       # embedding dimension for LTM/EM
    stm_max_turns: int = 20        # max turns to keep in STM
    ltm_max_entries: int = 1000    # max entries in LTM
    em_max_entries: int = 200      # max episodic summaries
    ltm_similarity_threshold: float = 0.4  # cosine sim threshold to store in LTM
    em_similarity_threshold: float = 0.5   # cosine sim threshold for episodes
    normalize_embeddings: bool = True      # whether to L2-normalize stored vectors


@dataclass
class STMMemoryEntry:
    """One short-term memory entry (raw text-level)."""
    turn_id: int
    role: str                 # e.g., "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LTMMemoryEntry:
    """One long-term memory entry with vector embedding."""
    embedding: np.ndarray      # shape (dim,)
    text: str                  # human-readable representation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EMMemoryEntry:
    """One episodic memory entry (summary) with vector embedding."""
    embedding: np.ndarray      # shape (dim,)
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def _cosine_similarities(
    query: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between a query vector and a matrix of vectors.

    query: shape (dim,)
    matrix: shape (N, dim)
    returns: shape (N,)
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query must be 1D (dim,)")

    q = query
    if np.linalg.norm(q) == 0.0:
        # avoid divide-by-zero; return zeros
        return np.zeros((matrix.shape[0],), dtype=np.float32)

    q = q / (np.linalg.norm(q) + 1e-9)
    m = matrix
    # assume matrix is already normalized if needed
    sims = m @ q  # (N,)
    return sims.astype(np.float32)


# ---------------------------------------------------------------------------
# Memory module
# ---------------------------------------------------------------------------

class MemoryModule:
    """
    Hierarchical memory for agentic RAG.

    Responsibilities:
      - Maintain STM as a fixed-length buffer of recent dialog turns
      - Maintain LTM as vector store of important facts
      - Maintain EM as vector store of episode summaries
      - Provide `load_context(...)` to supply memory to the agent
      - Provide `store_interaction(...)` to update memory after each step
    """

    def __init__(self, config: MemoryConfig):
        self.config = config

        # Short-Term Memory: list of STMMemoryEntry
        self._stm: List[STMMemoryEntry] = []

        # Long-Term Memory: list of LTMMemoryEntry + stacked embeddings
        self._ltm_entries: List[LTMMemoryEntry] = []
        self._ltm_embeddings: Optional[np.ndarray] = None  # shape (N, dim)

        # Episodic Memory: list of EMMemoryEntry + stacked embeddings
        self._em_entries: List[EMMemoryEntry] = []
        self._em_embeddings: Optional[np.ndarray] = None    # shape (N, dim)

        self._turn_counter: int = 0

        logger.info(
            "Initialized MemoryModule(dim=%d, stm_max_turns=%d, ltm_max_entries=%d, em_max_entries=%d)",
            config.dim,
            config.stm_max_turns,
            config.ltm_max_entries,
            config.em_max_entries,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_context(
        self,
        query_embedding: np.ndarray,
        top_k_stm: int = 4,
        top_k_ltm: int = 4,
        top_k_em: int = 2,
    ) -> Dict[str, Any]:
        """
        Retrieve context from STM, LTM, and EM given a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding vector, shape (dim,) or (1, dim).
        top_k_stm : int
            Number of STM entries to return (most recent).
        top_k_ltm : int
            Max number of LTM entries to return (most similar).
        top_k_em : int
            Max number of EM entries to return (most similar).

        Returns
        -------
        Dict[str, Any]
            {
              "stm": List[STMMemoryEntry-like dicts],
              "ltm": List[LTMMemoryEntry-like dicts],
              "em": List[EMMemoryEntry-like dicts],
            }
        """
        if query_embedding.ndim == 2:
            if query_embedding.shape[0] != 1:
                raise ValueError("query_embedding must be (dim,) or (1, dim)")
            query_embedding = query_embedding[0]

        if query_embedding.shape[0] != self.config.dim:
            raise ValueError(
                f"Expected query_embedding dim={self.config.dim}, "
                f"got {query_embedding.shape[0]}"
            )

        with record_latency("context_load_memory", store_samples=True):
            context: Dict[str, Any] = {}

            # 1) STM: just take the most recent k turns
            with record_latency("context_load_memory.stm", store_samples=False):
                stm_context = self._load_stm(top_k_stm)

            # 2) LTM: similarity search over stored embeddings
            with record_latency("context_load_memory.ltm", store_samples=False):
                ltm_context = self._load_ltm(query_embedding, top_k_ltm)

            # 3) EM: similarity search over episodic embeddings
            with record_latency("context_load_memory.em", store_samples=False):
                em_context = self._load_em(query_embedding, top_k_em)

            context["stm"] = stm_context
            context["ltm"] = ltm_context
            context["em"] = em_context

        return context

    def store_interaction(
        self,
        role: str,
        content: str,
        query_embedding: Optional[np.ndarray] = None,
        ltm_candidate_embedding: Optional[np.ndarray] = None,
        ltm_candidate_text: Optional[str] = None,
        ltm_metadata: Optional[Dict[str, Any]] = None,
        em_candidate_embedding: Optional[np.ndarray] = None,
        em_summary: Optional[str] = None,
        em_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update STM/LTM/EM after an agent step or a turn.

        Typical usage pattern:
          - Always store STM (role, content).
          - Optionally provide `ltm_candidate_embedding` + text/metadata to
            let the module decide whether to promote it into LTM.
          - Optionally provide `em_candidate_embedding` + summary to let
            the module update episodic memory at episode boundaries.

        Parameters
        ----------
        role : str
            "user", "assistant", "system", etc.
        content : str
            Raw text of the turn (may be prompt, answer, tool output).
        query_embedding : Optional[np.ndarray]
            Embedding for the query / interaction, used for gating LTM/EM.
        ltm_candidate_embedding : Optional[np.ndarray]
            Candidate embedding for LTM entry (shape (dim,) or (1, dim)).
        ltm_candidate_text : Optional[str]
            Human-readable representation for LTM (fact, key sentence, etc.).
        ltm_metadata : Optional[Dict[str, Any]]
            Additional metadata for LTM.
        em_candidate_embedding : Optional[np.ndarray]
            Candidate embedding for episodic summary (shape (dim,) or (1, dim)).
        em_summary : Optional[str]
            Text summary of the episode (or partial episode).
        em_metadata : Optional[Dict[str, Any]]
            Additional metadata for episodes.
        """
        with record_latency("context_store_memory", store_samples=True):
            self._turn_counter += 1
            turn_id = self._turn_counter

            # 1) Update STM
            self._update_stm(turn_id, role, content)

            # 2) Optionally update LTM
            if ltm_candidate_embedding is not None and ltm_candidate_text is not None:
                self._maybe_update_ltm(
                    query_embedding=query_embedding,
                    candidate_embedding=ltm_candidate_embedding,
                    candidate_text=ltm_candidate_text,
                    metadata=ltm_metadata or {},
                )

            # 3) Optionally update EM
            if em_candidate_embedding is not None and em_summary is not None:
                self._maybe_update_em(
                    query_embedding=query_embedding,
                    candidate_embedding=em_candidate_embedding,
                    summary=em_summary,
                    metadata=em_metadata or {},
                )

    # ------------------------------------------------------------------ #
    # STM implementation
    # ------------------------------------------------------------------ #

    def _update_stm(self, turn_id: int, role: str, content: str) -> None:
        entry = STMMemoryEntry(
            turn_id=turn_id,
            role=role,
            content=content,
        )
        self._stm.append(entry)
        # enforce capacity (keep the most recent stm_max_turns)
        if len(self._stm) > self.config.stm_max_turns:
            overflow = len(self._stm) - self.config.stm_max_turns
            if overflow > 0:
                self._stm = self._stm[overflow:]

    def _load_stm(self, top_k: int) -> List[Dict[str, Any]]:
        if top_k <= 0 or not self._stm:
            return []
        # get the most recent top_k turns
        entries = self._stm[-top_k:]
        # return as plain dicts
        return [
            {
                "turn_id": e.turn_id,
                "role": e.role,
                "content": e.content,
                "metadata": dict(e.metadata),
            }
            for e in entries
        ]

    # ------------------------------------------------------------------ #
    # LTM implementation
    # ------------------------------------------------------------------ #

    def _maybe_update_ltm(
        self,
        query_embedding: Optional[np.ndarray],
        candidate_embedding: np.ndarray,
        candidate_text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Decide whether to insert candidate into LTM based on similarity
        to query and existing LTM contents.
        """
        cand = self._ensure_dim(candidate_embedding)

        if self.config.normalize_embeddings:
            cand = _l2_normalize(cand)[0]
        else:
            cand = cand[0]

        # gating based on similarity with query
        if query_embedding is not None:
            q = self._ensure_dim(query_embedding)
            if self.config.normalize_embeddings:
                q = _l2_normalize(q)[0]
            else:
                q = q[0]
            sim = float(np.dot(cand, q) / (np.linalg.norm(cand) * np.linalg.norm(q) + 1e-9))
            if sim < self.config.ltm_similarity_threshold:
                # not relevant enough; do not store
                return

        # if LTM is empty, just add
        if self._ltm_embeddings is None or self._ltm_embeddings.shape[0] == 0:
            self._ltm_embeddings = cand[None, :]
            self._ltm_entries.append(
                LTMMemoryEntry(embedding=cand, text=candidate_text, metadata=metadata)
            )
            self._enforce_ltm_capacity()
            return

        # optionally avoid duplicates: check similarity with existing LTM
        sims = _cosine_similarities(cand, self._ltm_embeddings)
        if sims.size > 0 and float(np.max(sims)) >= 0.99:
            # near-duplicate; skip
            return

        # append
        self._ltm_embeddings = np.vstack([self._ltm_embeddings, cand[None, :]])
        self._ltm_entries.append(
            LTMMemoryEntry(embedding=cand, text=candidate_text, metadata=metadata)
        )
        self._enforce_ltm_capacity()

    def _enforce_ltm_capacity(self) -> None:
        if len(self._ltm_entries) <= self.config.ltm_max_entries:
            return
        overflow = len(self._ltm_entries) - self.config.ltm_max_entries
        if overflow <= 0:
            return
        # drop the oldest entries
        self._ltm_entries = self._ltm_entries[overflow:]
        if self._ltm_embeddings is not None:
            self._ltm_embeddings = self._ltm_embeddings[overflow:, :]

    def _load_ltm(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0 or self._ltm_embeddings is None or self._ltm_embeddings.shape[0] == 0:
            return []

        q = self._ensure_dim(query_embedding)
        if self.config.normalize_embeddings:
            q = _l2_normalize(q)[0]
        else:
            q = q[0]

        sims = _cosine_similarities(q, self._ltm_embeddings)  # shape (N,)
        k = min(top_k, sims.shape[0])
        if k <= 0:
            return []

        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            entry = self._ltm_entries[idx]
            results.append(
                {
                    "score": float(sims[idx]),
                    "text": entry.text,
                    "metadata": dict(entry.metadata),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # EM implementation
    # ------------------------------------------------------------------ #

    def _maybe_update_em(
        self,
        query_embedding: Optional[np.ndarray],
        candidate_embedding: np.ndarray,
        summary: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Decide whether to insert candidate into Episodic Memory based on
        similarity to query / existing episodes. This is typically called
        at episode boundaries with a summary embedding.
        """
        cand = self._ensure_dim(candidate_embedding)

        if self.config.normalize_embeddings:
            cand = _l2_normalize(cand)[0]
        else:
            cand = cand[0]

        if query_embedding is not None:
            q = self._ensure_dim(query_embedding)
            if self.config.normalize_embeddings:
                q = _l2_normalize(q)[0]
            else:
                q = q[0]
            sim = float(np.dot(cand, q) / (np.linalg.norm(cand) * np.linalg.norm(q) + 1e-9))
            if sim < self.config.em_similarity_threshold:
                # not sufficiently tied to the query/task; skip
                return

        # if EM is empty, just add
        if self._em_embeddings is None or self._em_embeddings.shape[0] == 0:
            self._em_embeddings = cand[None, :]
            self._em_entries.append(
                EMMemoryEntry(embedding=cand, summary=summary, metadata=metadata)
            )
            self._enforce_em_capacity()
            return

        # optionally avoid near-duplicates in EM
        sims = _cosine_similarities(cand, self._em_embeddings)
        if sims.size > 0 and float(np.max(sims)) >= 0.99:
            return

        self._em_embeddings = np.vstack([self._em_embeddings, cand[None, :]])
        self._em_entries.append(
            EMMemoryEntry(embedding=cand, summary=summary, metadata=metadata)
        )
        self._enforce_em_capacity()

    def _enforce_em_capacity(self) -> None:
        if len(self._em_entries) <= self.config.em_max_entries:
            return
        overflow = len(self._em_entries) - self.config.em_max_entries
        if overflow <= 0:
            return
        self._em_entries = self._em_entries[overflow:]
        if self._em_embeddings is not None:
            self._em_embeddings = self._em_embeddings[overflow:, :]

    def _load_em(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0 or self._em_embeddings is None or self._em_embeddings.shape[0] == 0:
            return []

        q = self._ensure_dim(query_embedding)
        if self.config.normalize_embeddings:
            q = _l2_normalize(q)[0]
        else:
            q = q[0]

        sims = _cosine_similarities(q, self._em_embeddings)  # shape (N,)
        k = min(top_k, sims.shape[0])
        if k <= 0:
            return []

        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            entry = self._em_entries[idx]
            results.append(
                {
                    "score": float(sims[idx]),
                    "summary": entry.summary,
                    "metadata": dict(entry.metadata),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def _ensure_dim(self, vec: np.ndarray) -> np.ndarray:
        """Ensure a vector is shape (1, dim)."""
        if vec.ndim == 1:
            if vec.shape[0] != self.config.dim:
                raise ValueError(
                    f"Expected 1D vec dim={self.config.dim}, got {vec.shape[0]}"
                )
            return vec[None, :]
        elif vec.ndim == 2:
            if vec.shape[1] != self.config.dim:
                raise ValueError(
                    f"Expected 2D vec shape (*, {self.config.dim}), got {vec.shape}"
                )
            if vec.shape[0] != 1:
                raise ValueError("Expected vec shape (1, dim) when 2D.")
            return vec
        else:
            raise ValueError("vec must be 1D or 2D ndarray.")

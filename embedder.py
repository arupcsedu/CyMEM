"""
embedder.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 18/10/2025

Embedding utilities for Deep RC RAG.

This module provides a thin wrapper around HuggingFace transformer
models for generating dense vector embeddings for both:
  - corpus chunks (documents)
  - queries (user questions / agent prompts)

It also records latency metrics for:
  - embedding_corpus
  - embedding_query

via the global metrics recorder in `metrics.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from metrics import record_latency


logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"  # "cpu" or "cuda"
    max_length: int = 512
    batch_size: int = 32
    normalize: bool = True  # L2-normalize embeddings


class Embedder:
    """
    Wrapper around a HuggingFace encoder model for generating embeddings.

    The class assumes a standard encoder architecture (BERT-like). It uses
    mean-pooling over the last hidden state of non-padding tokens to produce
    a fixed-size embedding for each input text.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config

        logger.info("Loading encoder model '%s' on device=%s",
                    config.model_name, config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Fix pad_token warnings for models like GPT-2
        if self.tokenizer.pad_token_id is None:
            logger.warning(
                "Tokenizer for %s has no pad_token_id; "
                "using eos_token as pad_token.",
                config.model_name,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(config.model_name)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def embed_corpus(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of corpus chunks.

        Parameters
        ----------
        texts : List[str]
            List of input strings (document chunks).

        Returns
        -------
        np.ndarray
            Array of shape (N, D) where N = len(texts).
        """
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        logger.info("Embedding corpus: %d chunks", len(texts))
        all_embeddings: List[np.ndarray] = []

        with record_latency("embedding_corpus", store_samples=True):
            for batch_texts in _batch_iter(texts, self.config.batch_size):
                batch_emb = self._embed_batch(batch_texts)
                all_embeddings.append(batch_emb)

        embeddings = np.vstack(all_embeddings)
        if self.config.normalize and embeddings.size > 0:
            embeddings = _l2_normalize(embeddings)

        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """
        Compute an embedding for a single query string.

        Parameters
        ----------
        text : str
            Input query.

        Returns
        -------
        np.ndarray
            1D array of shape (D,).
        """
        if not text:
            return np.zeros((self.model.config.hidden_size,), dtype=np.float32)

        with record_latency("embedding_query", store_samples=True):
            emb_batch = self._embed_batch([text])

        emb = emb_batch[0]
        if self.config.normalize:
            emb = _l2_normalize(emb[None, :])[0]
        return emb

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a small batch of texts into embeddings.

        Returns np.ndarray with shape (batch_size, hidden_dim).
        """
        # 1) Tokenize
        with record_latency("embedding.tokenize"):
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2) Forward pass
        with torch.no_grad(), record_latency("embedding.forward"):
            outputs = self.model(**inputs)

        # We assume outputs.last_hidden_state is available (B, T, H)
        last_hidden_state = outputs.last_hidden_state  # type: ignore[attr-defined]
        attention_mask = inputs.get("attention_mask")

        # 3) Mean pool over non-pad tokens
        embeddings = _mean_pooling(last_hidden_state, attention_mask)
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        return embeddings


# ---------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------- #

def _batch_iter(items: List[str], batch_size: int):
    """Yield successive batches from a list."""
    n = len(items)
    for i in range(0, n, batch_size):
        yield items[i : i + batch_size]


def _mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Mean-pool the last hidden state over non-padding tokens.

    Parameters
    ----------
    last_hidden_state : torch.Tensor
        (batch_size, seq_len, hidden_dim)
    attention_mask : Optional[torch.Tensor]
        (batch_size, seq_len), where 1 indicates real tokens.

    Returns
    -------
    torch.Tensor
        (batch_size, hidden_dim)
    """
    if attention_mask is None:
        # Simple mean over sequence dimension
        return last_hidden_state.mean(dim=1)

    # Expand mask to match hidden size
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    # Sum over tokens and divide by count of non-pad tokens
    masked_sum = (last_hidden_state * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)
    return masked_sum / token_counts


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
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms

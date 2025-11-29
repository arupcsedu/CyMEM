"""
agents.py

Agent orchestration for Deep RC RAG.

This module defines:

  - LLMConfig / LLMGenerator:
      thin wrapper around a HuggingFace causal language model (e.g., GPT-2)
      for answer generation.

  - AgentConfig:
      controls prompt formatting and retrieval sizes.

  - RagAgent:
      orchestrates:
        * query embedding
        * vector store retrieval (RAG)
        * memory retrieval (STM / LTM / EM)
        * context fusion
        * LLM answer generation
        * memory updates after each interaction

Latency metrics:

  - context_build_total
      wraps the entire context-building step:
        embed_query + vectorstore.search + memory.load_context

  - llm.generate
      wraps the model.generate() call inside LLMGenerator.

All lower-level latencies from embedder, vectorstore, and memory
are recorded in their respective modules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import record_latency
from embedder import Embedder
from vectorstore import VectorStore
from memory import MemoryModule


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model_name: str = "gpt2"
    device: str = "cpu"           # "cpu" or "cuda"
    max_input_length: int = 512   # for tokenization truncation
    max_new_tokens: int = 256     # length of generated continuation
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True        # sampling vs greedy


class LLMGenerator:
    """
    Thin wrapper around a causal language model (e.g., GPT-2).

    It is careful to:
      - set `pad_token` if the tokenizer lacks one (e.g., GPT-2),
      - use `truncation=True` when max_length is set for inputs,
      - pass only `max_new_tokens` (not `max_length`) to `generate()`
        to avoid HF warnings.

    In addition, it:
      - wraps generation with a `llm.generate` latency metric
      - tracks total generated tokens and generation time for throughput stats.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

        logger.info("Loading LLM '%s' on device=%s",
                    config.model_name, config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Fix pad_token warnings for GPT-2-like models
        if self.tokenizer.pad_token_id is None:
            logger.warning(
                "Tokenizer for %s has no pad_token_id; using eos_token as pad_token.",
                config.model_name,
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

        # Aggregated generation stats
        self.total_generated_tokens: int = 0
        self.total_generation_time_s: float = 0.0
        self.num_generations: int = 0

    def generate(
        self,
        prompt: str,
        extra_context: Optional[str] = None,
        **_: Any,
    ) -> str:
        """
        Generate an answer given a prompt and optional context string.

        The full input to the model is:

            [prompt]\n\n[extra_context (if any)]

        Tokenization:
          - truncation=True
          - max_length = self.config.max_input_length

        Generation:
          - max_new_tokens = self.config.max_new_tokens
          - do_sample = self.config.do_sample
          - temperature, top_p as configured

        Extra keyword arguments (**_) are accepted and ignored to remain
        compatible with different agent call signatures.
        """
        if extra_context:
            full_prompt = f"{prompt}\n\n{extra_context}"
        else:
            full_prompt = prompt

        # 1) Tokenize with truncation for inputs
        inputs = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt",
        ).to(self.device)

        # 2) Configure sampling flags
        do_sample = self.config.do_sample and self.config.temperature is not None

        # 3) Generate WITHOUT max_length (only max_new_tokens)
        start = time.perf_counter()
        with torch.no_grad():
            with record_latency("llm.generate", store_samples=True):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.config.temperature if do_sample else 1.0,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        elapsed = time.perf_counter() - start

        # 4) Token accounting: input vs total → generated
        generated_ids = outputs[0]
        input_len = inputs["input_ids"].shape[1]
        total_len = generated_ids.shape[0]
        num_generated = max(total_len - input_len, 0)

        self.total_generated_tokens += num_generated
        self.total_generation_time_s += elapsed
        self.num_generations += 1

        # Option A: decode only the newly generated tokens
        continuation = self.tokenizer.decode(
            generated_ids[input_len:],
            skip_special_tokens=True,
        )
        return continuation.strip()

    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Return aggregate stats about LLM generation:
          - total_generated_tokens
          - total_generation_time_s
          - num_generations
          - avg_latency_per_generation_s
          - tokens_per_second
        """
        if self.num_generations == 0 or self.total_generation_time_s <= 0:
            return {
                "total_generated_tokens": 0,
                "total_generation_time_s": 0.0,
                "num_generations": 0,
                "avg_latency_per_generation_s": None,
                "tokens_per_second": None,
            }

        avg_gen_time = self.total_generation_time_s / self.num_generations
        tokens_per_second = (
            self.total_generated_tokens / self.total_generation_time_s
            if self.total_generation_time_s > 0
            else None
        )

        return {
            "total_generated_tokens": self.total_generated_tokens,
            "total_generation_time_s": self.total_generation_time_s,
            "num_generations": self.num_generations,
            "avg_latency_per_generation_s": avg_gen_time,
            "tokens_per_second": tokens_per_second,
        }


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """
    Configuration for the RagAgent.

    Parameters
    ----------
    system_prompt : str
        High-level instruction describing agent behavior.
    top_k_vectorstore : int
        How many neighbors to retrieve from the vector store.
    top_k_stm : int
        How many recent turns to retrieve from STM.
    top_k_ltm : int
        How many matches to retrieve from LTM.
    top_k_em : int
        How many episodic summaries to retrieve from EM.
    """
    system_prompt: str = (
        "You are an AI assistant that answers questions using the provided context. "
        "Always ground your answer in the retrieved documents and memory."
    )
    top_k_vectorstore: int = 5
    top_k_stm: int = 4
    top_k_ltm: int = 4
    top_k_em: int = 2


# ---------------------------------------------------------------------------
# RAG + Memory agent
# ---------------------------------------------------------------------------

class RagAgent:
    """
    Agent that orchestrates:

      - query embedding via `Embedder`
      - vectorstore retrieval
      - memory retrieval (STM / LTM / EM)
      - context fusion
      - LLM answer generation
      - memory updates after each interaction

    It also records a higher-level latency metric:

      - context_build_total

    wrapping the entire "build context" step, which includes:
      * embedder.embed_query(query)
      * vectorstore.search(query_embedding, top_k_vectorstore)
      * memory.load_context(query_embedding, top_k_*).
    """

    def __init__(
        self,
        embedder: Embedder,
        vectorstore: VectorStore,
        memory: MemoryModule,
        llm: LLMGenerator,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.memory = memory
        self.llm = llm
        self.config = config or AgentConfig()

        logger.info(
            "Initialized RagAgent(top_k_vectorstore=%d, top_k_stm=%d, "
            "top_k_ltm=%d, top_k_em=%d)",
            self.config.top_k_vectorstore,
            self.config.top_k_stm,
            self.config.top_k_ltm,
            self.config.top_k_em,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_context(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Build the full context for the given query by:

          1) embedding the query
          2) retrieving from vectorstore
          3) retrieving from memory (STM/LTM/EM)
          4) formatting a context string for the LLM

        We wrap the entire sequence in a `context_build_total` latency
        metric, while the lower-level modules record their own internal
        timing (embedding, vectorstore, memory).
        """
        with record_latency("context_build_total", store_samples=True):
            # 1) Query embedding
            q_emb = self.embedder.embed_query(query)

            # 2) Vector store retrieval
            vs_results = self.vectorstore.search(
                query_embedding=q_emb,
                top_k=self.config.top_k_vectorstore,
            )

            # 3) Hierarchical memory retrieval
            memory_context = self.memory.load_context(
                query_embedding=q_emb,
                top_k_stm=self.config.top_k_stm,
                top_k_ltm=self.config.top_k_ltm,
                top_k_em=self.config.top_k_em,
            )

            # 4) Format a context string for the LLM
            context_str = self._format_context(
                query=query,
                vs_results=vs_results,
                memory_context=memory_context,
            )

        debug_info = {
            "query_embedding": q_emb,
            "vectorstore_results": vs_results,
            "memory_context": memory_context,
        }
        return context_str, debug_info

    def generate_answer(
        self,
        query: str,
        store_in_memory: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        High-level method: build context + generate answer + update memory.

        Returns
        -------
        answer : str
            The generated answer from the LLM.
        debug : Dict[str, Any]
            Extra info (context, retrieval results, etc.) useful for logging.
        """
        # 1) Build context (instrumented by context_build_total)
        context_str, debug_info = self.build_context(query)

        # 2) Construct final prompt to LLM
        prompt = self._build_prompt(query, context_str)

        # 3) Generate answer from the LLM
        answer = self.llm.generate(
            prompt=prompt,
            extra_context=None,   # context already in prompt
        )

        debug_info["prompt"] = prompt
        debug_info["answer"] = answer

        # 4) Update memory if requested
        if store_in_memory:
            self._update_memory_after_answer(
                query=query,
                answer=answer,
                debug_info=debug_info,
            )

        return answer, debug_info

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_prompt(self, query: str, context_str: str) -> str:
        """
        Build a full prompt for the LLM by combining:

          - system prompt
          - retrieved context
          - user query
        """
        prompt = (
            f"{self.config.system_prompt}\n\n"
            f"=== Retrieved Context ===\n"
            f"{context_str}\n\n"
            f"=== User Question ===\n"
            f"{query}\n\n"
            f"=== Answer ==="
        )
        return prompt

    def _format_context(
        self,
        query: str,
        vs_results: List[Dict[str, Any]],
        memory_context: Dict[str, Any],
    ) -> str:
        """
        Turn vectorstore results + memory entries into a single readable
        context string for the LLM.

        The structure is:

            [VectorStore Context]
            [Short-Term Memory]
            [Long-Term Memory]
            [Episodic Memory]
        """
        parts: List[str] = []

        # VectorStore context
        if vs_results:
            parts.append("VectorStore Results:")
            for i, r in enumerate(vs_results, start=1):
                score = r.get("score", 0.0)
                text = r.get("text", "")
                parts.append(f"[VS {i} | score={score:.3f}] {text}")
            parts.append("")  # blank line

        # STM context
        stm_entries = memory_context.get("stm", [])
        if stm_entries:
            parts.append("Short-Term Memory (recent turns):")
            for e in stm_entries:
                role = e.get("role", "unknown")
                content = e.get("content", "")
                parts.append(f"[STM {role}] {content}")
            parts.append("")

        # LTM context
        ltm_entries = memory_context.get("ltm", [])
        if ltm_entries:
            parts.append("Long-Term Memory (relevant facts):")
            for i, e in enumerate(ltm_entries, start=1):
                score = e.get("score", 0.0)
                text = e.get("text", "")
                parts.append(f"[LTM {i} | score={score:.3f}] {text}")
            parts.append("")

        # EM context
        em_entries = memory_context.get("em", [])
        if em_entries:
            parts.append("Episodic Memory (summaries of past sessions):")
            for i, e in enumerate(em_entries, start=1):
                score = e.get("score", 0.0)
                summary = e.get("summary", "")
                parts.append(f"[EM {i} | score={score:.3f}] {summary}")
            parts.append("")

        # If no context at all
        if not parts:
            parts.append("No prior context available.")

        return "\n".join(parts)

    def _update_memory_after_answer(
        self,
        query: str,
        answer: str,
        debug_info: Dict[str, Any],
    ) -> None:
        """
        Update STM/LTM/EM after generating an answer.

        Strategy:
          - Always add (role="user", content=query) and
            (role="assistant", content=answer) to STM.
          - Promote the full answer as one LTM candidate.
          - Episodic updates can be performed periodically (e.g., every N turns)
            by providing an episodic embedding and summary.
        """
        # 1) Get query embedding to drive gating in LTM/EM
        q_emb = debug_info.get("query_embedding", None)
        if q_emb is None:
            # just recompute if missing
            q_emb = self.embedder.embed_query(query)
            debug_info["query_embedding"] = q_emb

        # 2) Store user query in STM
        self.memory.store_interaction(
            role="user",
            content=query,
            query_embedding=q_emb,
        )

        # 3) Treat the full answer as 1 LTM candidate for now
        ltm_candidate_text = answer
        ltm_candidate_emb = self.embedder.embed_query(ltm_candidate_text)

        # 4) Store assistant answer in STM + optionally LTM
        self.memory.store_interaction(
            role="assistant",
            content=answer,
            query_embedding=q_emb,
            ltm_candidate_embedding=ltm_candidate_emb,
            ltm_candidate_text=ltm_candidate_text,
            ltm_metadata={"source": "agent_answer"},
            # em_candidate_embedding=None,
            # em_summary=None,
        )

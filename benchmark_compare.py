#!/usr/bin/env python3
"""
benchmark_compare.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 22/12/2025

Side-by-side benchmarking for Deep RC RAG:
  - Baseline: every rank preprocess+embed+index redundantly
  - Global: rank0 builds index once, other ranks only load+eval

This script:
  1) Loads metrics JSON files from two run directories (baseline vs global)
  2) Aggregates metrics across ranks (if multiple rank files exist)
  3) Computes per-run:
       - end-to-end runtime (robust heuristic)
       - docs processed, chunks produced
       - throughput: docs/sec, chunks/sec
  4) Exports:
       - compare_table.csv
       - compare_table.json
       - speedup.png (optional, requires matplotlib)
       - speedup.csv

Expected inputs (either):
  - a single latencies.json in each run dir, OR
  - multiple rank files like latencies_rank0.json, latencies_rank1.json, ...
  - optionally throughput.json, llm_generation_stats.json, etc.

Usage:
  python benchmark_compare.py \
      --baseline_dir runs/baseline_8ranks \
      --global_dir   runs/global_8ranks \
      --out_dir      runs/compare_out

Notes:
  - If matplotlib is not installed, the script will still export CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Helpers: IO
# ----------------------------

def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ----------------------------
# Metrics parsing
# ----------------------------

def _discover_metrics_jsons(run_dir: Path) -> List[Path]:
    """
    Prefer rank-suffixed latencies JSONs if present, otherwise fall back to latencies.json.
    """
    candidates = []
    # rank files
    candidates.extend(sorted(run_dir.glob("latencies_rank*.json")))
    candidates.extend(sorted(run_dir.glob("latency_rank*.json")))
    candidates.extend(sorted(run_dir.glob("metrics_rank*.json")))

    if candidates:
        return candidates

    # single file fallback
    single = run_dir / "latencies.json"
    if single.exists():
        return [single]
    single2 = run_dir / "latency.json"
    if single2.exists():
        return [single2]

    # last resort: any json with "latenc" or "metric"
    wild = sorted(run_dir.glob("*.json"))
    filtered = [p for p in wild if ("latenc" in p.name.lower() or "metric" in p.name.lower())]
    return filtered


def _extract_metric_entry(entry: Any) -> Tuple[int, float]:
    """
    Normalize one metric entry into (count, total_time_s).

    We support a few likely shapes:
      A) {"count": N, "total_time_s": T, ...}
      B) {"count": N, "total_time": T, ...}
      C) {"samples": [...], "count": N, "sum_s": T, ...}
    """
    if not isinstance(entry, dict):
        return 0, 0.0

    count = 0
    total_s = 0.0

    # common keys
    if "count" in entry:
        count = _safe_int(entry.get("count"))
    elif "n" in entry:
        count = _safe_int(entry.get("n"))

    for k in ("total_time_s", "total_s", "sum_s", "total_time", "sum", "total"):
        if k in entry:
            total_s = _safe_float(entry.get(k))
            break

    # if samples exist but totals not present, attempt derive
    if total_s == 0.0 and isinstance(entry.get("samples"), list):
        # samples could be in seconds or ms; assume seconds if small
        s = entry["samples"]
        if s:
            # heuristic: if median > 10, likely ms; convert to s
            vals = [_safe_float(v) for v in s]
            vals = [v for v in vals if v >= 0]
            if vals:
                mid = sorted(vals)[len(vals) // 2]
                if mid > 10.0:  # likely ms
                    total_s = sum(vals) / 1000.0
                else:
                    total_s = sum(vals)

    return count, total_s


def _load_metrics_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a metrics JSON file and normalize into:
      metrics[key] = {"count": int, "total_time_s": float, ...}

    If the file already has that shape, we keep it.
    """
    raw = _read_json(path)

    # Some dumps store under a top-level key
    # e.g., {"metrics": {...}} or {"_entries": {...}}
    if isinstance(raw, dict) and "metrics" in raw and isinstance(raw["metrics"], dict):
        raw = raw["metrics"]
    elif isinstance(raw, dict) and "_entries" in raw and isinstance(raw["_entries"], dict):
        raw = raw["_entries"]

    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out

    for k, v in raw.items():
        c, t = _extract_metric_entry(v)
        if c == 0 and t == 0.0 and isinstance(v, dict):
            # still store, but normalized fields help
            out[k] = dict(v)
            out[k].setdefault("count", 0)
            out[k].setdefault("total_time_s", 0.0)
        else:
            out[k] = dict(v) if isinstance(v, dict) else {}
            out[k]["count"] = c
            out[k]["total_time_s"] = t

    return out


def aggregate_run_metrics(run_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """
    Aggregate metrics across rank files. Returns:
      (aggregated_metrics, num_files_used)
    """
    files = _discover_metrics_jsons(run_dir)
    agg: Dict[str, Dict[str, Any]] = {}

    for fp in files:
        m = _load_metrics_file(fp)
        for key, ent in m.items():
            c, t = _extract_metric_entry(ent)
            if key not in agg:
                agg[key] = {"count": 0, "total_time_s": 0.0}
            agg[key]["count"] += c
            agg[key]["total_time_s"] += t

    return agg, len(files)


# ----------------------------
# Runtime + throughput heuristics
# ----------------------------

def _pick_docs_chunks_counts(metrics: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    """
    Estimate total docs and chunks processed from metric counts.
    """
    doc_candidates = [
        "preprocessing_documents",
        "preprocess.clean",
        "preprocess.chunk",
    ]
    chunk_candidates = [
        "preprocess.chunk_per_chunk",
        "chunk_per_chunk",
    ]

    docs = 0
    for k in doc_candidates:
        if k in metrics:
            docs = max(docs, _safe_int(metrics[k].get("count", 0)))
    # preprocess.load is per-file (not per-doc) in your design, so prefer per-doc
    if docs == 0 and "preprocess.load" in metrics:
        docs = _safe_int(metrics["preprocess.load"].get("count", 0))

    chunks = 0
    for k in chunk_candidates:
        if k in metrics:
            chunks = max(chunks, _safe_int(metrics[k].get("count", 0)))

    # fallback: if no chunk_per_chunk, approximate by vectorstore add count
    if chunks == 0:
        for k in ("context_store_vectorstore", "vectorstore.add_documents", "vectorstore.add", "add_documents"):
            if k in metrics:
                chunks = max(chunks, _safe_int(metrics[k].get("count", 0)))

    return docs, chunks


def estimate_end_to_end_runtime_s(metrics: Dict[str, Dict[str, Any]]) -> float:
    """
    Robust heuristic:
      1) If a metric looks like an end-to-end total run, prefer it.
      2) Else sum major stages if present.
      3) Else fall back to max(total_time_s) among all metrics (conservative).
    """
    # Strong candidates (you referenced preprocessing_total_run in your updated preprocessing.py header)
    preferred_keys = [
        "pipeline_total_run",
        "end_to_end_total",
        "total_run",
        "main_total_run",
        "preprocessing_total_run",
        "preprocessing_documents_total",  # sometimes used
        "preprocessing_documents",        # if you stored total_time as per-doc and aggregated
    ]

    for k in preferred_keys:
        if k in metrics:
            t = _safe_float(metrics[k].get("total_time_s", 0.0))
            if t > 0:
                return t

    # Compose from major phases if available
    stage_keys = [
        "preprocessing_total_run",
        "embed_corpus_total",
        "embedder.embed_corpus_total",
        "context_store_vectorstore",
        "vectorstore.build_index_total",
        "vectorstore.add_documents_total",
        "eval_total_run",
        "context_build_total",  # if eval exists, this can be significant
        "llm_generate_total",
    ]
    total = 0.0
    found = False
    for k in stage_keys:
        if k in metrics:
            t = _safe_float(metrics[k].get("total_time_s", 0.0))
            if t > 0:
                total += t
                found = True
    if found and total > 0:
        return total

    # Conservative fallback: max single metric total
    mx = 0.0
    for ent in metrics.values():
        mx = max(mx, _safe_float(ent.get("total_time_s", 0.0)))
    return mx


# ----------------------------
# Output structures
# ----------------------------

@dataclass
class RunSummary:
    name: str
    run_dir: str
    metrics_files_used: int
    docs: int
    chunks: int
    runtime_s: float
    docs_per_sec: float
    chunks_per_sec: float


def summarize_run(name: str, run_dir: Path) -> RunSummary:
    metrics, nfiles = aggregate_run_metrics(run_dir)
    docs, chunks = _pick_docs_chunks_counts(metrics)
    runtime_s = estimate_end_to_end_runtime_s(metrics)

    docs_per_sec = (docs / runtime_s) if runtime_s > 0 else 0.0
    chunks_per_sec = (chunks / runtime_s) if runtime_s > 0 else 0.0

    return RunSummary(
        name=name,
        run_dir=str(run_dir),
        metrics_files_used=nfiles,
        docs=docs,
        chunks=chunks,
        runtime_s=runtime_s,
        docs_per_sec=docs_per_sec,
        chunks_per_sec=chunks_per_sec,
    )


# ----------------------------
# Export: table + plots
# ----------------------------

def export_compare_csv(rows: List[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run",
            "run_dir",
            "metrics_files_used",
            "docs",
            "chunks",
            "runtime_s",
            "docs_per_sec",
            "chunks_per_sec",
        ])
        for r in rows:
            w.writerow([
                r.name, r.run_dir, r.metrics_files_used,
                r.docs, r.chunks,
                f"{r.runtime_s:.6f}",
                f"{r.docs_per_sec:.6f}",
                f"{r.chunks_per_sec:.6f}",
            ])


def export_compare_json(rows: List[RunSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)


def print_compare_table(rows: List[RunSummary]) -> None:
    # Simple pretty print
    def fmt(r: RunSummary) -> str:
        return (
            f"{r.name:>10} | docs={r.docs:>6} | chunks={r.chunks:>6} | "
            f"runtime={r.runtime_s:>9.3f}s | docs/s={r.docs_per_sec:>9.2f} | "
            f"chunks/s={r.chunks_per_sec:>9.2f} | files={r.metrics_files_used}"
        )

    print("\n=== Benchmark Comparison (Baseline vs Global) ===")
    for r in rows:
        print(fmt(r))

    # speedup (baseline / global)
    if len(rows) >= 2:
        b = rows[0]
        g = rows[1]
        if g.runtime_s > 0:
            speedup = b.runtime_s / g.runtime_s
        else:
            speedup = 0.0
        print(f"\nSpeedup (baseline/global) = {speedup:.3f}×")


def export_speedup_csv(baseline: RunSummary, global_: RunSummary, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    speedup = (baseline.runtime_s / global_.runtime_s) if global_.runtime_s > 0 else 0.0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "global", "speedup_x"])
        w.writerow(["runtime_s", f"{baseline.runtime_s:.6f}", f"{global_.runtime_s:.6f}", f"{speedup:.6f}"])
        w.writerow(["docs_per_sec", f"{baseline.docs_per_sec:.6f}", f"{global_.docs_per_sec:.6f}",
                    f"{(global_.docs_per_sec / baseline.docs_per_sec) if baseline.docs_per_sec > 0 else 0.0:.6f}"])
        w.writerow(["chunks_per_sec", f"{baseline.chunks_per_sec:.6f}", f"{global_.chunks_per_sec:.6f}",
                    f"{(global_.chunks_per_sec / baseline.chunks_per_sec) if baseline.chunks_per_sec > 0 else 0.0:.6f}"])


def plot_speedup(baseline: RunSummary, global_: RunSummary, out_path: Path) -> None:
    """
    Creates a small bar chart:
      - runtime (lower is better)
      - throughput docs/s, chunks/s (higher is better)
    Requires matplotlib; if missing, we skip.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[plot] matplotlib not installed; skipping plot. (pip/conda install matplotlib)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["runtime_s (↓)", "docs/s (↑)", "chunks/s (↑)"]
    base_vals = [baseline.runtime_s, baseline.docs_per_sec, baseline.chunks_per_sec]
    glob_vals = [global_.runtime_s, global_.docs_per_sec, global_.chunks_per_sec]

    x = list(range(len(labels)))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], base_vals, width, label="baseline")
    plt.bar([i + width/2 for i in x], glob_vals, width, label="global")

    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Value")
    plt.title("Baseline vs Global-Index: End-to-End Runtime and Throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=str, required=True, help="Directory containing baseline run outputs")
    p.add_argument("--global_dir", type=str, required=True, help="Directory containing global-index run outputs")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for comparison artifacts")
    p.add_argument("--baseline_name", type=str, default="baseline", help="Label for baseline run")
    p.add_argument("--global_name", type=str, default="global", help="Label for global run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.baseline_dir)
    glob_dir = Path(args.global_dir)
    out_dir = Path(args.out_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"baseline_dir not found: {base_dir}")
    if not glob_dir.exists():
        raise FileNotFoundError(f"global_dir not found: {glob_dir}")

    baseline = summarize_run(args.baseline_name, base_dir)
    global_ = summarize_run(args.global_name, glob_dir)

    rows = [baseline, global_]
    print_compare_table(rows)

    export_compare_csv(rows, out_dir / "compare_table.csv")
    export_compare_json(rows, out_dir / "compare_table.json")
    export_speedup_csv(baseline, global_, out_dir / "speedup.csv")
    plot_speedup(baseline, global_, out_dir / "speedup.png")

    print(f"\nWrote:\n  {out_dir/'compare_table.csv'}\n  {out_dir/'compare_table.json'}\n  {out_dir/'speedup.csv'}\n  {out_dir/'speedup.png'} (if matplotlib installed)\n")


if __name__ == "__main__":
    main()

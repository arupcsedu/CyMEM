# metrics.py
'''
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 21/10/2025
'''
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager
import csv
import json
import logging

#pip install matplotlib


@dataclass
class MetricEntry:
    count: int = 0
    total: float = 0.0
    samples: List[float] = field(default_factory=list)

    def observe(self, value: float, store_samples: bool = False) -> None:
        self.count += 1
        self.total += value
        if store_samples:
            self.samples.append(value)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def min(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def max(self) -> float:
        return max(self.samples) if self.samples else 0.0


class MetricsRecorder:
    """Simple, thread-safe in-process latency recorder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, MetricEntry] = {}

    def observe(self, name: str, duration_sec: float, *, store_samples: bool = False) -> None:
        with self._lock:
            if name not in self._entries:
                self._entries[name] = MetricEntry()
            self._entries[name].observe(duration_sec, store_samples=store_samples)

    def summary(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                name: {
                    "count": entry.count,
                    "total_sec": entry.total,
                    "avg_ms": entry.avg * 1000.0,
                    "min_ms": entry.min * 1000.0,
                    "max_ms": entry.max * 1000.0,
                }
                for name, entry in self._entries.items()
            }

    def dump_csv(self, path: str) -> None:
        data = self.summary()
        if not data:
            logging.warning("MetricsRecorder.dump_csv(%s): no data to write", path)
            return
        fieldnames = ["name", "count", "total_sec", "avg_ms", "min_ms", "max_ms"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, stats in data.items():
                writer.writerow({"name": name, **stats})

    def dump_json(self, path: str) -> None:
        data = self.summary()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def log_summary(self, logger: Optional[logging.Logger] = None) -> None:
        if logger is None:
            logger = logging.getLogger(__name__)
        logger.info("=== Latency Metrics Summary ===")
        for name, stats in self.summary().items():
            logger.info(
                "%s: count=%d, avg=%.2f ms, total=%.3f s (min=%.2f ms, max=%.2f ms)",
                name,
                stats["count"],
                stats["avg_ms"],
                stats["total_sec"],
                stats["min_ms"],
                stats["max_ms"],
            )


# convenient global singleton
metrics = MetricsRecorder()


@contextmanager
def record_latency(name: str, *, store_samples: bool = False):
    """
    Context manager for timing a block.

    Example:
        with record_latency("preprocessing"):
            do_work()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        metrics.observe(name, end - start, store_samples=store_samples)

# ---------------------------------------------------------------------------
# Throughput summary utilities
# ---------------------------------------------------------------------------

def _get_samples(recorder, key: str):
    """
    Helper to robustly extract `samples` list from recorder._entries[key],
    regardless of whether entries are dicts or dataclass-like objects.
    """
    entries = getattr(recorder, "_entries", {})
    entry = entries.get(key)
    if entry is None:
        return []

    # Case 1: dict-like
    if isinstance(entry, dict):
        return entry.get("samples", [])

    # Case 2: object with attribute
    return getattr(entry, "samples", [])

# ---------------------------------------------------------------------------
# Internal helper: compute throughput from recorder
# ---------------------------------------------------------------------------

def _compute_throughput(recorder):
    """
    Compute document-level and chunk-level throughput stats from the recorder.

    Returns
    -------
    stats : dict
        {
          "docs": {
            "count": int,
            "avg_latency_s": float or None,
            "throughput_per_s": float or None,
          },
          "chunks": {
            "count": int,
            "avg_latency_s": float or None,
            "throughput_per_s": float or None,
          },
        }
    """
    # Document-level
    doc_key = "preprocessing_documents"
    doc_samples = _get_samples(recorder, doc_key)

    if doc_samples:
        avg_doc_time = sum(doc_samples) / len(doc_samples)
        docs_per_sec = 1.0 / avg_doc_time if avg_doc_time > 0 else float("inf")
        doc_stats = {
            "count": len(doc_samples),
            "avg_latency_s": avg_doc_time,
            "throughput_per_s": docs_per_sec,
        }
    else:
        doc_stats = {
            "count": 0,
            "avg_latency_s": None,
            "throughput_per_s": None,
        }

    # Chunk-level
    chunk_key = "preprocess.chunk_per_chunk"
    chunk_samples = _get_samples(recorder, chunk_key)

    if chunk_samples:
        avg_chunk_time = sum(chunk_samples) / len(chunk_samples)
        chunks_per_sec = 1.0 / avg_chunk_time if avg_chunk_time > 0 else float("inf")
        chunk_stats = {
            "count": len(chunk_samples),
            "avg_latency_s": avg_chunk_time,
            "throughput_per_s": chunks_per_sec,
        }
    else:
        chunk_stats = {
            "count": 0,
            "avg_latency_s": None,
            "throughput_per_s": None,
        }

    return {"docs": doc_stats, "chunks": chunk_stats}


def summarize_throughput(recorder):
    """
    Pretty-print throughput metrics (docs/sec and chunks/sec) for the console.
    """
    stats = _compute_throughput(recorder)

    print("\n=== Preprocessing Throughput Summary ===")

    # Docs
    d = stats["docs"]
    if d["count"] > 0 and d["avg_latency_s"] is not None:
        print(f"Documents processed: {d['count']}")
        print(f"Avg doc latency: {d['avg_latency_s'] * 1000:.2f} ms")
        print(f"Docs per second: {d['throughput_per_s']:.2f}")
    else:
        print("No document-level metrics found (preprocessing_documents).")

    print()

    # Chunks
    c = stats["chunks"]
    if c["count"] > 0 and c["avg_latency_s"] is not None:
        print(f"Chunks processed: {c['count']}")
        print(f"Avg chunk latency: {c['avg_latency_s'] * 1000:.2f} ms")
        print(f"Chunks per second: {c['throughput_per_s']:.2f}")
    else:
        print("No per-chunk metrics found (preprocess.chunk_per_chunk).")

    print("=======================================\n")

# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

import json
from pathlib import Path


def export_throughput_json(recorder, path: str | Path = "throughput.json"):
    """
    Export document/chunk throughput statistics to a JSON file.

    Parameters
    ----------
    recorder : MetricsRecorder
        The global or local metrics recorder instance.
    path : str or Path
        Output JSON file path.
    """
    stats = _compute_throughput(recorder)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[metrics] Throughput JSON written to {out_path}")

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

import csv


def export_throughput_csv(recorder, path: str | Path = "throughput.csv"):
    """
    Export document/chunk throughput statistics to a CSV file.

    The CSV has one row per level: "docs" and "chunks".

    Columns:
      level, count, avg_latency_s, throughput_per_s
    """
    stats = _compute_throughput(recorder)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for level in ("docs", "chunks"):
        s = stats[level]
        rows.append(
            {
                "level": level,
                "count": s["count"],
                "avg_latency_s": "" if s["avg_latency_s"] is None else s["avg_latency_s"],
                "throughput_per_s": "" if s["throughput_per_s"] is None else s["throughput_per_s"],
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["level", "count", "avg_latency_s", "throughput_per_s"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[metrics] Throughput CSV written to {out_path}")


# ---------------------------------------------------------------------------
# Matplotlib plot
# ---------------------------------------------------------------------------

def plot_throughput_matplotlib(recorder, path: str | Path | None = None):
    """
    Plot a simple bar chart of docs/sec and chunks/sec using matplotlib.

    Parameters
    ----------
    recorder : MetricsRecorder
        The metrics recorder instance.
    path : str or Path or None
        If provided, saves the figure to this file. If None, shows the plot.
    """
    import matplotlib.pyplot as plt

    stats = _compute_throughput(recorder)

    labels = []
    values = []

    d = stats["docs"]
    if d["throughput_per_s"] is not None:
        labels.append("docs/sec")
        values.append(d["throughput_per_s"])

    c = stats["chunks"]
    if c["throughput_per_s"] is not None:
        labels.append("chunks/sec")
        values.append(c["throughput_per_s"])

    if not labels:
        print("[metrics] No throughput data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Throughput (items / second)")
    ax.set_title("Preprocessing Throughput")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if path is not None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[metrics] Throughput plot saved to {out_path}")
        plt.close(fig)
    else:
        plt.show()



#!/usr/bin/env python
"""
download_hf_dataset.py
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 01/10/2025

Utility script to download a text dataset from Hugging Face and export it as
a folder of .txt files suitable for RAG preprocessing.

Usage examples:

  # Minimal: wikitext-2 train split, auto-detect text column
  python download_hf_dataset.py \
      --dataset_name wikitext \
      --subset_name wikitext-2-raw-v1 \
      --split train \
      --output_dir data/wikitext2_train

  # AG News: use 'text' column, limit to 10k samples
  python download_hf_dataset.py \
      --dataset_name ag_news \
      --split train \
      --text_column text \
      --max_samples 10000 \
      --output_dir data/ag_news_train
      
  python download_hf_dataset.py \
  --dataset_name wikitext \
  --subset_name wikitext-103-raw-v1 \
  --split train \
  --output_dir data/wikitext103_train

Requirements:
  pip install datasets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, List

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and export it as .txt files."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset on the Hub, e.g. 'wikitext', 'ag_news', 'squad'.",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default=None,
        help="Optional subset/config name, e.g. 'wikitext-2-raw-v1'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download, e.g. 'train', 'validation', 'test'.",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help=(
            "Name of the text column to export. If not provided, the script will "
            "try common names: 'text', 'content', 'article', 'document', 'body', "
            "or fall back to the first string column."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on number of samples (for smaller test corpora).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where .txt files will be written.",
    )

    return parser.parse_args()


def detect_text_column(dataset, explicit_column: Optional[str] = None) -> str:
    """
    Detect the text column to use.

    Priority:
      1) explicit_column if provided and exists
      2) any of the common names
      3) first column of type string
    """
    cols: List[str] = list(dataset.column_names)
    if explicit_column is not None:
        if explicit_column not in cols:
            raise ValueError(
                f"Requested text_column='{explicit_column}' not in dataset columns: {cols}"
            )
        return explicit_column

    # Try common names
    common_candidates = ["text", "content", "article", "document", "body", "context"]
    for c in common_candidates:
        if c in cols:
            return c

    # Fallback: first string-like column
    first_str_col = None
    for c in cols:
        if dataset.features[c].dtype in ("string", "large_string"):
            first_str_col = c
            break

    if first_str_col is None:
        raise ValueError(
            f"Could not auto-detect a text column. Available columns: {cols}"
        )

    return first_str_col


def main() -> None:
    args = parse_args()

    # Create output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset '{args.dataset_name}' "
          f"{f'/{args.subset_name}' if args.subset_name else ''} "
          f"split='{args.split}' from Hugging Face...")

    ds = load_dataset(
        path=args.dataset_name,
        name=args.subset_name,
        split=args.split,
    )

    # Detect which column to use as text
    text_col = detect_text_column(ds, args.text_column)
    print(f"[INFO] Using text column: '{text_col}'")
    print(f"[INFO] Dataset size: {len(ds)} rows")

    # Optionally subsample
    if args.max_samples is not None and args.max_samples < len(ds):
        print(f"[INFO] Subsampling to max_samples={args.max_samples}")
        ds = ds.select(range(args.max_samples))

    # Export each row to a separate .txt file
    print(f"[INFO] Writing samples to: {out_dir}")
    count = 0
    for i, row in enumerate(ds):
        text = str(row[text_col]).strip()
        if not text:
            continue

        fname = out_dir / f"doc_{i:06d}.txt"
        fname.write_text(text, encoding="utf-8")
        count += 1

        if count % 1000 == 0:
            print(f"[INFO] Written {count} files so far...")

    print(f"[DONE] Wrote {count} text files to {out_dir}")


if __name__ == "__main__":
    main()

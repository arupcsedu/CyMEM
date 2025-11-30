'''
Author: Arup Sarker
Email: djy8hg@virginia.edu, arupcsedu@gmail.com
Date: 01/10/2025
'''
#!/usr/bin/env python

from datasets import load_dataset
from pathlib import Path


def download_wikitext_eval(output_dir="data/wikitext_eval", max_samples=None):
    """
    Download the Wikitext-2 test split and save each entry as a .txt file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading Wikitext-2-raw-v1 test split...")
    ds = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test"
    )

    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))

    print(f"[INFO] Writing evaluation dataset to: {out}")

    count = 0
    for i, row in enumerate(ds):
        text = str(row["text"]).strip()
        if not text:
            continue
        (out / f"eval_{i:06d}.txt").write_text(text, encoding="utf-8")
        count += 1

    print(f"[DONE] Wrote {count} eval text files to {output_dir}")

    return output_dir

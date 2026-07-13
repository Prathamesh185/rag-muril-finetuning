"""
select_chunks.py

Purpose:
Select representative chunks from Vikaspedia passages.

Pipeline:
1. Load chunk CSV
2. Remove duplicate chunks
3. Remove tiny chunks
4. Group by document
5. Uniformly sample chunks from each document
6. Save selected_chunks.csv
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = Path("data/chunks/vikaspedia_passages.csv")
OUTPUT_FILE = Path("data/chunks/selected_chunks.csv")

MIN_WORDS = 60
KEEP_RATIO = 0.45
MIN_KEEP = 2


# ==========================================================
# Load CSV
# ==========================================================

print(f"Loading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Original chunks : {len(df)}")


# ==========================================================
# Remove duplicates
# ==========================================================

before = len(df)

df = df.drop_duplicates(subset=["chunk_text"])

print(f"Removed duplicates : {before - len(df)}")


# ==========================================================
# Remove tiny chunks
# ==========================================================

before = len(df)

df = df[df["word_count"] >= MIN_WORDS]

print(f"Removed tiny chunks : {before - len(df)}")

print(f"Remaining chunks : {len(df)}")


# ==========================================================
# Group by article
# ==========================================================

groups = df.groupby("document_id")

selected = []

total_articles = len(groups)

print(f"\nProcessing {total_articles} articles...\n")


# ==========================================================
# Uniform sampling
# ==========================================================

for article_id, group in groups:

    group = group.sort_values("chunk_index")

    n = len(group)

    # Keep all very small articles
    if n <= MIN_KEEP:
        selected.append(group)
        continue

    k = max(
        MIN_KEEP,
        math.ceil(n * KEEP_RATIO)
    )

    indices = np.linspace(
        0,
        n - 1,
        k,
        dtype=int
    )

    selected.append(group.iloc[indices])


# ==========================================================
# Merge
# ==========================================================

selected_df = pd.concat(selected)

selected_df = selected_df.sort_values(
    ["document_id", "chunk_index"]
)

selected_df.reset_index(drop=True, inplace=True)


# ==========================================================
# Save
# ==========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

selected_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("=" * 50)
print("Selection Complete")
print("=" * 50)

print(f"Articles              : {total_articles}")
print(f"Selected chunks       : {len(selected_df)}")
print(f"Saved to              : {OUTPUT_FILE}")

print("\nSample:")

print(
    selected_df[
        [
            "document_id",
            "chunk_index",
            "word_count",
            "title"
        ]
    ].head(10)
)
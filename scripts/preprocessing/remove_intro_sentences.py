import re
import pandas as pd
from pathlib import Path

INPUT = Path("data/chunks/selected_chunks_fixed.csv")
OUTPUT = Path("data/chunks/selected_chunks_fixed.csv")   # overwrite


df = pd.read_csv(INPUT)

intro_patterns = [
    r"^इस\s+भाग\s+में.*?[।|.!?]",
    r"^इस\s+लेख\s+में.*?[।|.!?]",
    r"^इस\s+पृष्ठ\s+में.*?[।|.!?]",
]

removed = 0


def clean_intro(text):
    global removed

    if pd.isna(text):
        return text

    text = str(text).strip()

    original = text

    for pattern in intro_patterns:
        text = re.sub(
            pattern,
            "",
            text,
            count=1,
            flags=re.IGNORECASE | re.DOTALL
        ).strip()

    if text != original:
        removed += 1

    return text


df["chunk_text"] = df["chunk_text"].apply(clean_intro)

df.to_csv(INPUT, index=False, encoding="utf-8")

print("=" * 60)
print("Intro sentences removed :", removed)
print("Saved :", OUTPUT)
print("=" * 60)
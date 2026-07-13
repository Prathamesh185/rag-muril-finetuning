import pandas as pd
from pathlib import Path

INPUT = Path("data/chunks/selected_chunks.csv")
OUTPUT = Path("data/chunks/all_titles.csv")

df = pd.read_csv(INPUT)

titles = (
    df[["document_id", "title"]]
    .drop_duplicates()
    .sort_values("title")
)

titles.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8"
)

print("Total titles:", len(titles))
print("Saved:", OUTPUT)
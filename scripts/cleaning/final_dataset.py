import pandas as pd

INPUT_FILE = "data/cleaned/keep_stage3_final.csv"
OUTPUT_FILE = "data/cleaned/final_dataset.csv"

df = pd.read_csv(INPUT_FILE)

df = df[
    [
        "question",
        "positive",
        "pair_id",
        "chunk_id",
        "document_id",
        "title",
        "source",
        "url",
    ]
]

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("Saved:", OUTPUT_FILE)
print("Rows:", len(df))
print("Columns:", list(df.columns))
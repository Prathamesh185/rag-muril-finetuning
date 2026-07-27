import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = "data/cleaned/final_dataset.csv"
OUTPUT_DIR = "data/training/"

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("=" * 60)
print("Dataset Statistics")
print("=" * 60)
print(f"Total Question-Passage Pairs : {len(df):,}")
print(f"Unique Documents            : {df['document_id'].nunique():,}")

# ==========================================================
# DOCUMENT-LEVEL SPLIT
# ==========================================================

doc_ids = df["document_id"].unique()

np.random.seed(RANDOM_SEED)
np.random.shuffle(doc_ids)

n_docs = len(doc_ids)

train_end = int(TRAIN_RATIO * n_docs)
valid_end = int((TRAIN_RATIO + VALID_RATIO) * n_docs)

train_docs = set(doc_ids[:train_end])
valid_docs = set(doc_ids[train_end:valid_end])
test_docs = set(doc_ids[valid_end:])

train_df = (
    df[df["document_id"].isin(train_docs)]
    .reset_index(drop=True)
)

valid_df = (
    df[df["document_id"].isin(valid_docs)]
    .reset_index(drop=True)
)

test_df = (
    df[df["document_id"].isin(test_docs)]
    .reset_index(drop=True)
)

# ==========================================================
# VERIFY NO DOCUMENT LEAKAGE
# ==========================================================

assert len(train_docs & valid_docs) == 0
assert len(train_docs & test_docs) == 0
assert len(valid_docs & test_docs) == 0

print("\n No document overlap detected.")

# ==========================================================
# VERIFY ALL ROWS ACCOUNTED FOR
# ==========================================================

assert len(train_df) + len(valid_df) + len(test_df) == len(df)

print("All rows accounted for.")

# ==========================================================
# VERIFY ALL DOCUMENTS ACCOUNTED FOR
# ==========================================================

assert (
    train_df["document_id"].nunique()
    + valid_df["document_id"].nunique()
    + test_df["document_id"].nunique()
    == df["document_id"].nunique()
)

print("All documents accounted for.")

# ==========================================================
# PRINT SPLIT STATISTICS
# ==========================================================

print("\n" + "=" * 60)
print("Split Statistics")
print("=" * 60)

print(f"Train : {len(train_df):,} pairs | {train_df['document_id'].nunique():,} documents")
print(f"Valid : {len(valid_df):,} pairs | {valid_df['document_id'].nunique():,} documents")
print(f"Test  : {len(test_df):,} pairs | {test_df['document_id'].nunique():,} documents")

print("\nPair Distribution")
print("-" * 60)

print(f"Train : {len(train_df)/len(df):.2%}")
print(f"Valid : {len(valid_df)/len(df):.2%}")
print(f"Test  : {len(test_df)/len(df):.2%}")

print("\nDocument Distribution")
print("-" * 60)

print(f"Train : {len(train_docs)} ({len(train_docs)/n_docs:.2%})")
print(f"Valid : {len(valid_docs)} ({len(valid_docs)/n_docs:.2%})")
print(f"Test  : {len(test_docs)} ({len(test_docs)/n_docs:.2%})")

# ==========================================================
# SAVE SPLITS
# ==========================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

train_df.to_csv(
    f"{OUTPUT_DIR}/train.csv",
    index=False,
    encoding="utf-8-sig"
)

valid_df.to_csv(
    f"{OUTPUT_DIR}/valid.csv",
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    f"{OUTPUT_DIR}/test.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("Files Saved Successfully")
print("=" * 60)
print(f"✓ {OUTPUT_DIR}/train.csv")
print(f"✓ {OUTPUT_DIR}/valid.csv")
print(f"✓ {OUTPUT_DIR}/test.csv")
print("=" * 60)
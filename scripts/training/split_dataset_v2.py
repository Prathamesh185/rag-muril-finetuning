import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = "data/cleaned_v2/final_dataset_v2.csv"
OUTPUT_DIR = "data/training_v2"

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print("=" * 65)
print("Dataset V2 Statistics")
print("=" * 65)

print(
    f"Total Question-Passage Pairs : "
    f"{len(df):,}"
)

print(
    f"Unique Documents             : "
    f"{df['document_id'].nunique():,}"
)

print(
    f"Unique Split Groups          : "
    f"{df['split_group_id'].nunique():,}"
)

print(
    f"Unique Chunks                : "
    f"{df['chunk_id'].nunique():,}"
)


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = {
    "question",
    "positive",
    "pair_id",
    "chunk_id",
    "document_id",
    "split_group_id",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


# ==========================================================
# GROUP-LEVEL SPLIT
# ==========================================================

group_ids = (
    df["split_group_id"]
    .dropna()
    .unique()
)


np.random.seed(
    RANDOM_SEED
)

np.random.shuffle(
    group_ids
)


n_groups = len(
    group_ids
)


train_end = int(
    TRAIN_RATIO
    * n_groups
)

valid_end = int(
    (
        TRAIN_RATIO
        + VALID_RATIO
    )
    * n_groups
)


train_groups = set(
    group_ids[
        :train_end
    ]
)

valid_groups = set(
    group_ids[
        train_end:
        valid_end
    ]
)

test_groups = set(
    group_ids[
        valid_end:
    ]
)


# ==========================================================
# CREATE SPLITS
# ==========================================================

train_df = (
    df[
        df["split_group_id"]
        .isin(train_groups)
    ]
    .reset_index(drop=True)
)

valid_df = (
    df[
        df["split_group_id"]
        .isin(valid_groups)
    ]
    .reset_index(drop=True)
)

test_df = (
    df[
        df["split_group_id"]
        .isin(test_groups)
    ]
    .reset_index(drop=True)
)


# ==========================================================
# VERIFY NO GROUP LEAKAGE
# ==========================================================

assert (
    len(
        train_groups
        & valid_groups
    )
    == 0
)

assert (
    len(
        train_groups
        & test_groups
    )
    == 0
)

assert (
    len(
        valid_groups
        & test_groups
    )
    == 0
)


print(
    "\n✓ No split_group_id overlap detected."
)


# ==========================================================
# VERIFY NO DOCUMENT LEAKAGE
# ==========================================================

train_docs = set(
    train_df["document_id"]
)

valid_docs = set(
    valid_df["document_id"]
)

test_docs = set(
    test_df["document_id"]
)


assert (
    len(
        train_docs
        & valid_docs
    )
    == 0
)

assert (
    len(
        train_docs
        & test_docs
    )
    == 0
)

assert (
    len(
        valid_docs
        & test_docs
    )
    == 0
)


print(
    "✓ No document overlap detected."
)


# ==========================================================
# VERIFY NO PAIR OVERLAP
# ==========================================================

train_pairs = set(
    train_df["pair_id"]
)

valid_pairs = set(
    valid_df["pair_id"]
)

test_pairs = set(
    test_df["pair_id"]
)


assert not (
    train_pairs
    & valid_pairs
)

assert not (
    train_pairs
    & test_pairs
)

assert not (
    valid_pairs
    & test_pairs
)


print(
    "✓ No pair_id overlap detected."
)


# ==========================================================
# VERIFY NO CHUNK OVERLAP
# ==========================================================

train_chunks = set(
    train_df["chunk_id"]
)

valid_chunks = set(
    valid_df["chunk_id"]
)

test_chunks = set(
    test_df["chunk_id"]
)


assert not (
    train_chunks
    & valid_chunks
)

assert not (
    train_chunks
    & test_chunks
)

assert not (
    valid_chunks
    & test_chunks
)


print(
    "✓ No chunk_id overlap detected."
)


# ==========================================================
# VERIFY ALL ROWS ACCOUNTED FOR
# ==========================================================

assert (
    len(train_df)
    + len(valid_df)
    + len(test_df)
    == len(df)
)


print(
    "✓ All rows accounted for."
)


# ==========================================================
# VERIFY ALL GROUPS ACCOUNTED FOR
# ==========================================================

assert (
    len(train_groups)
    + len(valid_groups)
    + len(test_groups)
    == n_groups
)


print(
    "✓ All split groups accounted for."
)


# ==========================================================
# VERIFY ALL DOCUMENTS ACCOUNTED FOR
# ==========================================================

assert (
    train_df["document_id"].nunique()
    + valid_df["document_id"].nunique()
    + test_df["document_id"].nunique()
    == df["document_id"].nunique()
)


print(
    "✓ All documents accounted for."
)


# ==========================================================
# SPLIT STATISTICS
# ==========================================================

print(
    "\n"
    + "=" * 65
)

print(
    "Split Statistics"
)

print(
    "=" * 65
)


print(
    f"Train : "
    f"{len(train_df):,} pairs | "
    f"{train_df['document_id'].nunique():,} documents | "
    f"{train_df['split_group_id'].nunique():,} groups"
)

print(
    f"Valid : "
    f"{len(valid_df):,} pairs | "
    f"{valid_df['document_id'].nunique():,} documents | "
    f"{valid_df['split_group_id'].nunique():,} groups"
)

print(
    f"Test  : "
    f"{len(test_df):,} pairs | "
    f"{test_df['document_id'].nunique():,} documents | "
    f"{test_df['split_group_id'].nunique():,} groups"
)


# ==========================================================
# PAIR DISTRIBUTION
# ==========================================================

print(
    "\nPair Distribution"
)

print(
    "-" * 65
)


print(
    f"Train : "
    f"{len(train_df) / len(df):.2%}"
)

print(
    f"Valid : "
    f"{len(valid_df) / len(df):.2%}"
)

print(
    f"Test  : "
    f"{len(test_df) / len(df):.2%}"
)


# ==========================================================
# GROUP DISTRIBUTION
# ==========================================================

print(
    "\nSplit Group Distribution"
)

print(
    "-" * 65
)


print(
    f"Train : "
    f"{len(train_groups):,} "
    f"({len(train_groups) / n_groups:.2%})"
)

print(
    f"Valid : "
    f"{len(valid_groups):,} "
    f"({len(valid_groups) / n_groups:.2%})"
)

print(
    f"Test  : "
    f"{len(test_groups):,} "
    f"({len(test_groups) / n_groups:.2%})"
)


# ==========================================================
# DOCUMENT DISTRIBUTION
# ==========================================================

total_documents = (
    df["document_id"]
    .nunique()
)


print(
    "\nDocument Distribution"
)

print(
    "-" * 65
)


print(
    f"Train : "
    f"{train_df['document_id'].nunique():,} "
    f"({train_df['document_id'].nunique() / total_documents:.2%})"
)

print(
    f"Valid : "
    f"{valid_df['document_id'].nunique():,} "
    f"({valid_df['document_id'].nunique() / total_documents:.2%})"
)

print(
    f"Test  : "
    f"{test_df['document_id'].nunique():,} "
    f"({test_df['document_id'].nunique() / total_documents:.2%})"
)


# ==========================================================
# SAVE SPLITS
# ==========================================================

Path(
    OUTPUT_DIR
).mkdir(
    parents=True,
    exist_ok=True
)


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


# ==========================================================
# SAVE SPLIT GROUP MAPPING
# ==========================================================

group_split_rows = []


for group_id in train_groups:

    group_split_rows.append(
        {
            "split_group_id":
                group_id,
            "split":
                "train",
        }
    )


for group_id in valid_groups:

    group_split_rows.append(
        {
            "split_group_id":
                group_id,
            "split":
                "valid",
        }
    )


for group_id in test_groups:

    group_split_rows.append(
        {
            "split_group_id":
                group_id,
            "split":
                "test",
        }
    )


group_split_df = pd.DataFrame(
    group_split_rows
)


group_split_df.to_csv(
    f"{OUTPUT_DIR}/split_group_mapping.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print(
    "\n"
    + "=" * 65
)

print(
    "Files Saved Successfully"
)

print(
    "=" * 65
)


print(
    f"✓ {OUTPUT_DIR}/train.csv"
)

print(
    f"✓ {OUTPUT_DIR}/valid.csv"
)

print(
    f"✓ {OUTPUT_DIR}/test.csv"
)

print(
    f"✓ {OUTPUT_DIR}/split_group_mapping.csv"
)


print(
    "\nIMPORTANT:"
)

print(
    "These are the V2 leakage-safe splits."
)

print(
    "Use these files for retraining and final evaluation."
)

print(
    "Do not use the old data/training splits for Experiment V2."
)

print(
    "=" * 65
)
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

FULL_FILE = Path(
    "data/training_v2/train_hard_negatives_v2.csv"
)

REVIEW_FILE = Path(
    "data/training_v2/hard_negatives_review_v2_verified.csv"
)

OUTPUT_FILE = Path(
    "data/training_v2/train_hard_negatives_v2_clean.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CLEAN HARD NEGATIVES - V2")
print("=" * 60)

full_df = pd.read_csv(FULL_FILE)
review_df = pd.read_csv(REVIEW_FILE)

print(f"\nFull hard-negative rows: {len(full_df):,}")
print(f"Reviewed sample rows: {len(review_df):,}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_full_columns = {
    "pair_id",
    "negative_chunk_id",
}

required_review_columns = {
    "pair_id",
    "negative_chunk_id",
    "review_label",
}

missing_full = required_full_columns - set(full_df.columns)
missing_review = required_review_columns - set(review_df.columns)

if missing_full:
    raise ValueError(
        f"Missing columns in full hard-negative file: {missing_full}"
    )

if missing_review:
    raise ValueError(
        f"Missing columns in review file: {missing_review}"
    )


# ============================================================
# NORMALIZE IDENTIFIERS
# ============================================================

for df in [full_df, review_df]:
    df["pair_id"] = df["pair_id"].astype(str).str.strip()
    df["negative_chunk_id"] = (
        df["negative_chunk_id"]
        .astype(str)
        .str.strip()
    )

review_df["review_label"] = (
    review_df["review_label"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# IDENTIFY UNSAFE REVIEWED NEGATIVES
# ============================================================

# Conservative choice:
# FALSE_NEGATIVE -> definitely remove
# UNCERTAIN      -> remove because we cannot confidently
#                   say it is a true negative

unsafe_labels = {
    "FALSE_NEGATIVE",
    "UNCERTAIN",
}

unsafe_df = review_df[
    review_df["review_label"].isin(unsafe_labels)
].copy()

print("\nReviewed labels:")
print(review_df["review_label"].value_counts(dropna=False))

print(
    f"\nUnsafe reviewed negatives to remove: "
    f"{len(unsafe_df):,}"
)


# ============================================================
# CREATE COMPOSITE KEY
# ============================================================

full_df["_review_key"] = (
    full_df["pair_id"]
    + "|||"
    + full_df["negative_chunk_id"]
)

unsafe_df["_review_key"] = (
    unsafe_df["pair_id"]
    + "|||"
    + unsafe_df["negative_chunk_id"]
)

unsafe_keys = set(unsafe_df["_review_key"])


# ============================================================
# CHECK THAT REVIEWED ROWS EXIST IN FULL DATASET
# ============================================================

matched_keys = set(full_df["_review_key"]) & unsafe_keys
missing_keys = unsafe_keys - set(full_df["_review_key"])

print(
    f"Unsafe rows matched in full dataset: "
    f"{len(matched_keys):,}"
)

if missing_keys:
    print(
        f"WARNING: {len(missing_keys):,} reviewed unsafe rows "
        f"were not found in the full dataset."
    )

    for key in list(missing_keys)[:10]:
        print("  Missing:", key)


# ============================================================
# REMOVE UNSAFE NEGATIVES
# ============================================================

clean_df = full_df[
    ~full_df["_review_key"].isin(unsafe_keys)
].copy()

removed_df = full_df[
    full_df["_review_key"].isin(unsafe_keys)
].copy()


# ============================================================
# REMOVE HELPER COLUMN
# ============================================================

clean_df.drop(
    columns=["_review_key"],
    inplace=True
)

removed_df.drop(
    columns=["_review_key"],
    inplace=True
)


# ============================================================
# SAFETY DEDUPLICATION
# ============================================================

before_dedup = len(clean_df)

clean_df = clean_df.drop_duplicates(
    subset=[
        "pair_id",
        "negative_chunk_id",
    ],
    keep="first"
).reset_index(drop=True)

duplicates_removed = before_dedup - len(clean_df)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

clean_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# Optional audit file showing exactly what was removed
REMOVED_FILE = Path(
    "data/training_v2/"
    "hard_negatives_removed_after_review.csv"
)

removed_df.to_csv(
    REMOVED_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print(
    f"Original triplets:        {len(full_df):,}"
)

print(
    f"Review-based removed:     {len(removed_df):,}"
)

print(
    f"Duplicate triplets removed: {duplicates_removed:,}"
)

print(
    f"Final clean triplets:     {len(clean_df):,}"
)

print(
    f"Unique training queries:  "
    f"{clean_df['pair_id'].nunique():,}"
)

print("\nSaved cleaned dataset to:")
print(OUTPUT_FILE)

print("\nRemoved rows audit saved to:")
print(REMOVED_FILE)
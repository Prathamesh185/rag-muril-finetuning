import pandas as pd
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "output_v2"
    / "retrieval_examples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "output_v2"
)

SORTED_FILE = OUTPUT_DIR / "retrieval_examples_sorted.csv"
BEST_DEMO_FILE = OUTPUT_DIR / "best_demo_examples.csv"


# ==========================================================
# LOAD FILE
# ==========================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print(f"Total examples: {len(df):,}")


# ==========================================================
# CONVERT RANKS TO NUMBERS
# ==========================================================

def rank_to_number(rank):

    if pd.isna(rank):
        return 11

    rank = str(rank).strip()

    if rank == "Not in Top10":
        return 11

    try:
        return int(float(rank))
    except ValueError:
        return 11


df["Base Rank Numeric"] = (
    df["Base Rank"].apply(rank_to_number)
)

df["Fine-tuned Rank Numeric"] = (
    df["Fine-tuned Rank"].apply(rank_to_number)
)


# ==========================================================
# CALCULATE IMPROVEMENT
# ==========================================================

df["Rank Improvement"] = (
    df["Base Rank Numeric"]
    - df["Fine-tuned Rank Numeric"]
)


# ==========================================================
# CLASSIFY EXAMPLES
# ==========================================================

def classify(row):

    base_rank = row["Base Rank Numeric"]
    fine_rank = row["Fine-tuned Rank Numeric"]

    if base_rank == 11 and fine_rank == 1:
        return "Best Demo"

    if base_rank == 11 and fine_rank <= 3:
        return "Strong Improvement"

    if fine_rank < base_rank:
        return "Improved"

    if fine_rank == base_rank:
        return "Same"

    return "Worse"


df["Category"] = df.apply(
    classify,
    axis=1
)


# ==========================================================
# SORT
# ==========================================================

df = df.sort_values(
    by=[
        "Rank Improvement",
        "Fine-tuned Rank Numeric",
    ],
    ascending=[
        False,
        True,
    ]
).reset_index(drop=True)


# ==========================================================
# SAVE ALL SORTED RESULTS
# ==========================================================

df.to_csv(
    SORTED_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# SAVE BEST DEMO EXAMPLES
# ==========================================================

best_demo = df[
    df["Category"] == "Best Demo"
].copy()

best_demo.to_csv(
    BEST_DEMO_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("QUALITATIVE RESULT SORTING COMPLETE")
print("=" * 60)

print("\nCategory counts:")
print(df["Category"].value_counts())

print("\nTop improvements:")

print(
    df[
        [
            "Question",
            "Base Rank",
            "Fine-tuned Rank",
            "Rank Improvement",
            "Category",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nSaved:")
print(SORTED_FILE)
print(BEST_DEMO_FILE)
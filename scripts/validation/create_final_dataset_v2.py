import pandas as pd
from pathlib import Path


# ==========================================================
# CONFIG
# ==========================================================

DATASET_FILE = (
    "data/validation_v2/final_dataset_grouped.csv"
)

QUESTION_DUPLICATES_FILE = (
    "data/validation_v2/question_checks/"
    "near_duplicate_questions_reviewed.csv"
)

QUESTION_COPYING_FILE = (
    "data/validation_v2/question_checks/"
    "question_copying_candidates_reviewed.csv"
)

OUTPUT_DIR = Path(
    "data/cleaned_v2"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "final_dataset_v2.csv"
)

REMOVED_FILE = (
    OUTPUT_DIR / "removed_rows_v2.csv"
)

DUPLICATE_GROUPS_FILE = (
    OUTPUT_DIR / "question_duplicate_groups.csv"
)


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD FILES
# ==========================================================

print("=" * 65)
print("Loading files...")
print("=" * 65)

df = pd.read_csv(
    DATASET_FILE,
    encoding="utf-8-sig"
)

duplicate_df = pd.read_csv(
    QUESTION_DUPLICATES_FILE,
    encoding="utf-8-sig"
)

copy_df = pd.read_csv(
    QUESTION_COPYING_FILE,
    encoding="utf-8-sig"
)


print(
    f"Original pairs               : {len(df):,}"
)

print(
    f"Question duplicate candidates: {len(duplicate_df):,}"
)

print(
    f"Copying candidates           : {len(copy_df):,}"
)


# ==========================================================
# VERIFY REQUIRED COLUMNS
# ==========================================================

dataset_required = {
    "pair_id",
    "question",
    "positive",
    "chunk_id",
    "document_id",
    "split_group_id",
}

duplicate_required = {
    "pair_id_a",
    "pair_id_b",
    "review_decision",
}

copy_required = {
    "pair_id",
    "review_decision",
}


missing = (
    dataset_required
    - set(df.columns)
)

if missing:

    raise ValueError(
        f"Dataset missing columns: {missing}"
    )


missing = (
    duplicate_required
    - set(duplicate_df.columns)
)

if missing:

    raise ValueError(
        f"Question duplicate file missing columns: {missing}"
    )


missing = (
    copy_required
    - set(copy_df.columns)
)

if missing:

    raise ValueError(
        f"Copying file missing columns: {missing}"
    )


# ==========================================================
# NORMALIZE PAIR IDs
# ==========================================================

df["pair_id"] = (
    df["pair_id"]
    .astype(str)
    .str.strip()
)

duplicate_df["pair_id_a"] = (
    duplicate_df["pair_id_a"]
    .astype(str)
    .str.strip()
)

duplicate_df["pair_id_b"] = (
    duplicate_df["pair_id_b"]
    .astype(str)
    .str.strip()
)

copy_df["pair_id"] = (
    copy_df["pair_id"]
    .astype(str)
    .str.strip()
)


# ==========================================================
# REMOVAL TRACKING
# ==========================================================

removed_rows = []


def mark_for_removal(
    pair_id,
    reason,
):

    removed_rows.append(
        {
            "pair_id": str(pair_id),
            "removal_reason": reason,
        }
    )


# ==========================================================
# STEP 1
# HANDLE EXACT DUPLICATE QUESTIONS
# ==========================================================

print(
    "\nChecking exact duplicate questions..."
)


# Normalize question for exact duplicate checking

def normalize_question(text):

    text = str(text).lower().strip()

    text = " ".join(
        text.split()
    )

    return text


df["_normalized_question"] = (
    df["question"]
    .apply(normalize_question)
)


exact_duplicate_groups = (
    df[
        df.duplicated(
            "_normalized_question",
            keep=False,
        )
    ]
    .groupby(
        "_normalized_question"
    )
)


exact_removed = 0


for _, group in exact_duplicate_groups:

    if len(group) <= 1:
        continue

    # Keep first row
    keep_row = group.iloc[0]

    for i in range(
        1,
        len(group)
    ):

        pair_id = (
            group.iloc[i]["pair_id"]
        )

        mark_for_removal(
            pair_id,
            "EXACT_DUPLICATE_QUESTION",
        )

        exact_removed += 1


print(
    f"Exact duplicate rows marked : "
    f"{exact_removed:,}"
)


# ==========================================================
# STEP 2
# BUILD NEAR-DUPLICATE QUESTION GROUPS
# ==========================================================

print(
    "\nBuilding duplicate-question groups..."
)


confirmed_duplicates = (
    duplicate_df[
        duplicate_df[
            "review_decision"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        == "DUPLICATE_QUESTION"
    ]
    .copy()
)


print(
    f"Confirmed duplicate relationships : "
    f"{len(confirmed_duplicates):,}"
)


# ==========================================================
# UNION-FIND
# ==========================================================

parent = {}
rank = {}


def make_set(x):

    if x not in parent:

        parent[x] = x
        rank[x] = 0


def find(x):

    if parent[x] != x:

        parent[x] = find(
            parent[x]
        )

    return parent[x]


def union(a, b):

    make_set(a)
    make_set(b)

    root_a = find(a)
    root_b = find(b)

    if root_a == root_b:
        return

    if rank[root_a] < rank[root_b]:

        parent[root_a] = root_b

    elif rank[root_a] > rank[root_b]:

        parent[root_b] = root_a

    else:

        parent[root_b] = root_a
        rank[root_a] += 1


# ==========================================================
# CREATE CONNECTED GROUPS
# ==========================================================

for _, row in confirmed_duplicates.iterrows():

    pair_a = row["pair_id_a"]
    pair_b = row["pair_id_b"]

    union(
        pair_a,
        pair_b,
    )


question_groups = {}


for pair_id in parent:

    root = find(
        pair_id
    )

    question_groups.setdefault(
        root,
        []
    ).append(
        pair_id
    )


print(
    f"Connected duplicate groups : "
    f"{len(question_groups):,}"
)


# ==========================================================
# CREATE QUESTION GROUP REPORT
# ==========================================================

group_report = []

near_duplicate_removed = 0


for group_number, pair_ids in enumerate(
    question_groups.values(),
    start=1,
):

    group_id = (
        f"question_group_{group_number:04d}"
    )


    # ------------------------------------------------------
    # Find rows in original dataset
    # ------------------------------------------------------

    group_rows = (
        df[
            df["pair_id"]
            .isin(pair_ids)
        ]
        .copy()
    )


    if len(group_rows) == 0:
        continue


    # ------------------------------------------------------
    # CHOOSE REPRESENTATIVE
    #
    # Prefer shortest clear question.
    # This avoids keeping unnecessarily verbose duplicates.
    # ------------------------------------------------------

    group_rows[
        "_question_length"
    ] = (
        group_rows["question"]
        .astype(str)
        .str.len()
    )


    group_rows = (
        group_rows
        .sort_values(
            [
                "_question_length",
                "pair_id",
            ]
        )
    )


    keep_pair_id = (
        group_rows.iloc[0][
            "pair_id"
        ]
    )


    # ------------------------------------------------------
    # SAVE GROUP INFORMATION
    # ------------------------------------------------------

    for _, row in group_rows.iterrows():

        pair_id = row["pair_id"]

        is_kept = (
            pair_id
            == keep_pair_id
        )


        group_report.append(
            {
                "question_group_id":
                    group_id,

                "pair_id":
                    pair_id,

                "question":
                    row["question"],

                "chunk_id":
                    row["chunk_id"],

                "document_id":
                    row["document_id"],

                "split_group_id":
                    row["split_group_id"],

                "kept":
                    is_kept,
            }
        )


        if not is_kept:

            mark_for_removal(
                pair_id,
                "NEAR_DUPLICATE_QUESTION",
            )

            near_duplicate_removed += 1


print(
    f"Near-duplicate rows marked : "
    f"{near_duplicate_removed:,}"
)


# ==========================================================
# STEP 3
# HANDLE QUESTION → PASSAGE COPYING
# ==========================================================

print(
    "\nProcessing question-copying decisions..."
)


copy_df[
    "review_decision"
] = (
    copy_df[
        "review_decision"
    ]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ----------------------------------------------------------
# Clear copying problems
# ----------------------------------------------------------

bad_copy_decisions = {
    "SOURCE_QA_COPY",
    "GENERATED_COPYING_PROBLEM",
}


bad_copy_df = (
    copy_df[
        copy_df[
            "review_decision"
        ]
        .isin(
            bad_copy_decisions
        )
    ]
)


copy_removed = 0


for _, row in bad_copy_df.iterrows():

    mark_for_removal(
        row["pair_id"],
        row["review_decision"],
    )

    copy_removed += 1


print(
    f"Clear copying rows marked : "
    f"{copy_removed:,}"
)


# ----------------------------------------------------------
# UNCERTAIN ROWS
# ----------------------------------------------------------

uncertain_df = (
    copy_df[
        copy_df[
            "review_decision"
        ]
        == "UNCERTAIN"
    ]
)


print(
    f"Uncertain copying rows kept : "
    f"{len(uncertain_df):,}"
)


# ==========================================================
# STEP 4
# CREATE REMOVAL TABLE
# ==========================================================

removed_df = pd.DataFrame(
    removed_rows
)


if len(removed_df):

    # One pair may have multiple reasons.
    # Combine them rather than losing information.

    removed_df = (
        removed_df
        .groupby(
            "pair_id"
        )[
            "removal_reason"
        ]
        .apply(
            lambda x:
            " | ".join(
                sorted(
                    set(x)
                )
            )
        )
        .reset_index()
    )


print(
    f"\nUnique rows marked for removal : "
    f"{len(removed_df):,}"
)


# ==========================================================
# STEP 5
# VERIFY PAIR IDS EXIST
# ==========================================================

dataset_pair_ids = set(
    df["pair_id"]
)


unknown_pair_ids = (
    set(
        removed_df["pair_id"]
    )
    - dataset_pair_ids
)


if unknown_pair_ids:

    print(
        "\nWARNING:"
    )

    print(
        f"{len(unknown_pair_ids)} removal pair IDs "
        f"were not found in the dataset."
    )


# ==========================================================
# STEP 6
# CREATE CLEAN DATASET
# ==========================================================

remove_ids = set(
    removed_df["pair_id"]
)


clean_df = (
    df[
        ~df["pair_id"]
        .isin(remove_ids)
    ]
    .copy()
)


# Remove temporary column
clean_df = clean_df.drop(
    columns=[
        "_normalized_question"
    ],
    errors="ignore",
)


# ==========================================================
# STEP 7
# ADD REMOVED ROW DETAILS TO REPORT
# ==========================================================

if len(removed_df):

    removed_report = (
        removed_df
        .merge(
            df[
                [
                    "pair_id",
                    "question",
                    "positive",
                    "chunk_id",
                    "document_id",
                    "split_group_id",
                ]
            ],
            on="pair_id",
            how="left",
        )
    )

else:

    removed_report = pd.DataFrame()


# ==========================================================
# STEP 8
# VALIDATION
# ==========================================================

print(
    "\nValidating final dataset..."
)


# ----------------------------------------------------------
# No pair IDs duplicated
# ----------------------------------------------------------

duplicate_pair_ids = (
    clean_df[
        "pair_id"
    ]
    .duplicated()
    .sum()
)


if duplicate_pair_ids > 0:

    raise ValueError(
        f"Found {duplicate_pair_ids} "
        f"duplicate pair IDs."
    )


# ----------------------------------------------------------
# No missing important fields
# ----------------------------------------------------------

important_columns = [
    "question",
    "positive",
    "pair_id",
    "chunk_id",
    "document_id",
    "split_group_id",
]


missing_values = (
    clean_df[
        important_columns
    ]
    .isna()
    .sum()
    .sum()
)


if missing_values > 0:

    raise ValueError(
        f"Found {missing_values} missing values "
        f"in important columns."
    )


# ----------------------------------------------------------
# Verify row accounting
# ----------------------------------------------------------

assert (
    len(clean_df)
    + len(removed_report)
    == len(df)
)


print(
    "✓ Pair IDs valid"
)

print(
    "✓ Important fields present"
)

print(
    "✓ All rows accounted for"
)


# ==========================================================
# STEP 9
# SAVE FILES
# ==========================================================

clean_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


removed_report.to_csv(
    REMOVED_FILE,
    index=False,
    encoding="utf-8-sig",
)


group_report_df = pd.DataFrame(
    group_report
)


group_report_df.to_csv(
    DUPLICATE_GROUPS_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print(
    "\n"
    + "=" * 65
)

print(
    "FINAL DATASET V2 CREATED"
)

print(
    "=" * 65
)


print(
    f"Original pairs               : "
    f"{len(df):,}"
)

print(
    f"Exact duplicates removed     : "
    f"{exact_removed:,}"
)

print(
    f"Near-question duplicates     : "
    f"{near_duplicate_removed:,}"
)

print(
    f"Copying problems marked      : "
    f"{copy_removed:,}"
)

print(
    f"Unique rows actually removed : "
    f"{len(removed_report):,}"
)

print(
    f"Final pairs                  : "
    f"{len(clean_df):,}"
)


print(
    f"\nUnique chunks    : "
    f"{clean_df['chunk_id'].nunique():,}"
)

print(
    f"Unique documents : "
    f"{clean_df['document_id'].nunique():,}"
)

print(
    f"Split groups     : "
    f"{clean_df['split_group_id'].nunique():,}"
)


print(
    "\nSaved:"
)

print(
    f"Clean dataset:\n{OUTPUT_FILE}"
)

print(
    f"\nRemoved rows report:\n{REMOVED_FILE}"
)

print(
    f"\nQuestion duplicate groups:\n"
    f"{DUPLICATE_GROUPS_FILE}"
)


print(
    "\nIMPORTANT:"
)

print(
    "Do not overwrite the old dataset."
)

print(
    "Use final_dataset_v2.csv for the "
    "new train/valid/test split."
)

print(
    "=" * 65
)
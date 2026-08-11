import pandas as pd
from pathlib import Path


# ==========================================================
# CONFIG
# ==========================================================

DATASET_FILE = "data/cleaned/final_dataset.csv"

REVIEWED_DUPLICATES_FILE = (
    "data/validation_v2/near_duplicate_candidates_reviewed.csv"
)

OUTPUT_DIR = Path("data/validation_v2")

GROUPED_DATASET_FILE = (
    OUTPUT_DIR / "final_dataset_grouped.csv"
)

GROUP_MAPPING_FILE = (
    OUTPUT_DIR / "document_groups.csv"
)


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 65)
print("Loading dataset...")
print("=" * 65)

df = pd.read_csv(
    DATASET_FILE,
    encoding="utf-8-sig"
)

review_df = pd.read_csv(
    REVIEWED_DUPLICATES_FILE,
    encoding="utf-8-sig"
)

print(
    f"Question-passage pairs : {len(df):,}"
)

print(
    f"Unique documents       : "
    f"{df['document_id'].nunique():,}"
)

print(
    f"Reviewed candidate rows: "
    f"{len(review_df):,}"
)


# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

required_dataset_columns = {
    "document_id"
}

required_review_columns = {
    "document_a",
    "document_b",
    "confirmed_duplicate",
}


missing_dataset_columns = (
    required_dataset_columns
    - set(df.columns)
)

missing_review_columns = (
    required_review_columns
    - set(review_df.columns)
)


if missing_dataset_columns:

    raise ValueError(
        f"Missing columns in dataset: "
        f"{missing_dataset_columns}"
    )


if missing_review_columns:

    raise ValueError(
        f"Missing columns in reviewed duplicate file: "
        f"{missing_review_columns}"
    )


# ==========================================================
# NORMALIZE DOCUMENT IDS
# ==========================================================

df["document_id"] = (
    df["document_id"]
    .astype(str)
    .str.strip()
)

review_df["document_a"] = (
    review_df["document_a"]
    .astype(str)
    .str.strip()
)

review_df["document_b"] = (
    review_df["document_b"]
    .astype(str)
    .str.strip()
)


# ==========================================================
# KEEP ONLY CONFIRMED DUPLICATE PAIRS
# ==========================================================

confirmed_values = {
    "GROUP_DUPLICATE",
    "TRUE",
    "YES",
    "1"
}

confirmed_pairs = review_df[
    review_df["confirmed_duplicate"]
    .astype(str)
    .str.strip()
    .str.upper()
    .isin(confirmed_values)
].copy()


print(
    f"\nConfirmed duplicate candidate rows: "
    f"{len(confirmed_pairs):,}"
)


# ==========================================================
# UNION-FIND / DISJOINT SET
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
# CREATE SET FOR EVERY DOCUMENT
# ==========================================================

all_documents = (
    df["document_id"]
    .dropna()
    .astype(str)
    .unique()
)


for document_id in all_documents:

    make_set(
        document_id
    )


# ==========================================================
# UNION CONFIRMED DUPLICATE DOCUMENTS
# ==========================================================

unknown_documents = []


for _, row in confirmed_pairs.iterrows():

    doc_a = row["document_a"]
    doc_b = row["document_b"]


    if doc_a not in parent:

        unknown_documents.append(
            doc_a
        )

        continue


    if doc_b not in parent:

        unknown_documents.append(
            doc_b
        )

        continue


    union(
        doc_a,
        doc_b
    )


# ==========================================================
# WARNING FOR UNKNOWN DOCUMENT IDS
# ==========================================================

if unknown_documents:

    unknown_documents = sorted(
        set(
            unknown_documents
        )
    )

    print(
        "\nWARNING:"
    )

    print(
        f"{len(unknown_documents)} document IDs from "
        f"the reviewed file were not found in "
        f"final_dataset.csv"
    )

    for doc in unknown_documents[:10]:

        print(
            " -",
            doc
        )


# ==========================================================
# BUILD CONNECTED DOCUMENT GROUPS
# ==========================================================

groups = {}


for document_id in all_documents:

    root = find(
        document_id
    )

    groups.setdefault(
        root,
        []
    ).append(
        document_id
    )


# ==========================================================
# ASSIGN CLEAN GROUP IDS
# ==========================================================

sorted_groups = sorted(
    groups.values(),
    key=lambda x: (
        -len(x),
        x[0]
    )
)


document_to_group = {}

group_rows = []


for group_number, documents in enumerate(
    sorted_groups,
    start=1
):

    split_group_id = (
        f"group_{group_number:05d}"
    )

    group_size = len(
        documents
    )


    for document_id in sorted(
        documents
    ):

        document_to_group[
            document_id
        ] = split_group_id

        group_rows.append(
            {
                "split_group_id":
                    split_group_id,

                "document_id":
                    document_id,

                "group_size":
                    group_size,

                "is_duplicate_group":
                    group_size > 1
            }
        )


# ==========================================================
# ADD GROUP ID TO FULL DATASET
# ==========================================================

df["split_group_id"] = (
    df["document_id"]
    .map(
        document_to_group
    )
)


# ==========================================================
# VERIFY ALL DOCUMENTS RECEIVED A GROUP
# ==========================================================

missing_group_count = (
    df["split_group_id"]
    .isna()
    .sum()
)


if missing_group_count > 0:

    raise ValueError(
        f"{missing_group_count} rows did not "
        f"receive a split_group_id."
    )


# ==========================================================
# CREATE GROUP MAPPING DATAFRAME
# ==========================================================

group_mapping_df = pd.DataFrame(
    group_rows
)


# ==========================================================
# GROUP STATISTICS
# ==========================================================

total_groups = (
    group_mapping_df[
        "split_group_id"
    ]
    .nunique()
)

duplicate_groups = (
    group_mapping_df[
        group_mapping_df[
            "group_size"
        ] > 1
    ][
        "split_group_id"
    ]
    .nunique()
)

documents_in_duplicate_groups = (
    group_mapping_df[
        group_mapping_df[
            "group_size"
        ] > 1
    ][
        "document_id"
    ]
    .nunique()
)

largest_group_size = (
    group_mapping_df[
        "group_size"
    ]
    .max()
)


# ==========================================================
# SAVE FILES
# ==========================================================

df.to_csv(
    GROUPED_DATASET_FILE,
    index=False,
    encoding="utf-8-sig"
)

group_mapping_df.to_csv(
    GROUP_MAPPING_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# PRINT SUMMARY
# ==========================================================

print("\n" + "=" * 65)
print("DUPLICATE DOCUMENT GROUPING COMPLETE")
print("=" * 65)

print(
    f"Total documents              : "
    f"{len(all_documents):,}"
)

print(
    f"Total split groups           : "
    f"{total_groups:,}"
)

print(
    f"Duplicate document groups    : "
    f"{duplicate_groups:,}"
)

print(
    f"Documents in duplicate groups: "
    f"{documents_in_duplicate_groups:,}"
)

print(
    f"Largest duplicate group size : "
    f"{largest_group_size}"
)


print(
    "\nGrouped dataset saved to:"
)

print(
    GROUPED_DATASET_FILE
)


print(
    "\nDocument group mapping saved to:"
)

print(
    GROUP_MAPPING_FILE
)


# ==========================================================
# PRINT DUPLICATE GROUP EXAMPLES
# ==========================================================

duplicate_only = (
    group_mapping_df[
        group_mapping_df[
            "group_size"
        ] > 1
    ]
)


if len(
    duplicate_only
) > 0:

    print(
        "\nExample duplicate groups:"
    )


    example_group_ids = (
        duplicate_only[
            "split_group_id"
        ]
        .drop_duplicates()
        .head(5)
        .tolist()
    )


    for group_id in example_group_ids:

        group_docs = (
            duplicate_only[
                duplicate_only[
                    "split_group_id"
                ]
                == group_id
            ][
                "document_id"
            ]
            .tolist()
        )

        print(
            f"\n{group_id}"
        )

        for doc in group_docs:

            print(
                "  -",
                doc
            )


print(
    "\nNext step:"
)

print(
    "Run question duplicate / leakage checks "
    "on final_dataset_grouped.csv before "
    "creating the final train/valid/test split."
)

print("=" * 65)
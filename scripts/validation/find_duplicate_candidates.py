import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = "data/cleaned/final_dataset.csv"

OUTPUT_DIR = Path("data/validation_v2")

EXACT_FILE = OUTPUT_DIR / "exact_duplicates.csv"

CANDIDATE_FILE = (
    OUTPUT_DIR / "near_duplicate_candidates.csv"
)

MODEL_PATH = "models/base_muril"

SEMANTIC_THRESHOLD = 0.95

TOP_K = 5

BATCH_SIZE = 32


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def normalize_text(text):

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = text.lower()

    # Remove invisible Unicode characters
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")

    # Normalize punctuation
    text = re.sub(
        r"[^\w\s\u0900-\u097f]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 65)
print("Loading final dataset...")
print("=" * 65)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

print(f"Question-passage pairs : {len(df):,}")
print(
    f"Unique chunks          : "
    f"{df['chunk_id'].nunique():,}"
)
print(
    f"Unique documents       : "
    f"{df['document_id'].nunique():,}"
)


# ==========================================================
# CREATE UNIQUE PASSAGE TABLE
# ==========================================================

passages = (
    df[
        [
            "chunk_id",
            "document_id",
            "positive",
            "title",
            "url",
        ]
    ]
    .drop_duplicates(
        subset=["chunk_id"]
    )
    .reset_index(drop=True)
)

passages["normalized_positive"] = (
    passages["positive"]
    .apply(normalize_text)
)

print(
    "\nUnique passage rows:",
    len(passages)
)


# ==========================================================
# EXACT DUPLICATE DETECTION
# ==========================================================

print("\nChecking exact duplicates...")

duplicate_mask = passages.duplicated(
    subset=["normalized_positive"],
    keep=False,
)

exact_duplicates = passages[
    duplicate_mask
].copy()

exact_duplicates.to_csv(
    EXACT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(
    "Exact duplicate rows:",
    len(exact_duplicates)
)


# ==========================================================
# REMOVE EXACT DUPLICATES FROM SEARCH TABLE
# ==========================================================

# Keep only one copy for semantic search.
# We are NOT deleting anything from final_dataset.csv yet.

search_passages = (
    passages
    .drop_duplicates(
        subset=["normalized_positive"],
        keep="first",
    )
    .reset_index(drop=True)
)

print(
    "Passages for semantic search:",
    len(search_passages)
)


# ==========================================================
# LOAD BASE MuRIL
# ==========================================================

print("\nLoading Base MuRIL...")

model = SentenceTransformer(
    MODEL_PATH
)


# ==========================================================
# CREATE EMBEDDINGS
# ==========================================================

print("\nEncoding passages...")

embeddings = model.encode(
    search_passages[
        "positive"
    ].tolist(),

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True,
)

embeddings = embeddings.astype(
    "float32"
)


# ==========================================================
# BUILD FAISS INDEX
# ==========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print(
    f"\nIndexed passages: "
    f"{index.ntotal:,}"
)


# ==========================================================
# SEARCH NEAREST PASSAGES
# ==========================================================

print("\nSearching near duplicates...")

scores, indices = index.search(
    embeddings,
    TOP_K + 1
)


# ==========================================================
# LEXICAL SIMILARITY
# ==========================================================

def lexical_similarity(text1, text2):

    return SequenceMatcher(
        None,
        normalize_text(text1),
        normalize_text(text2),
    ).ratio()


# ==========================================================
# BUILD CANDIDATE LIST
# ==========================================================

candidates = []

seen_pairs = set()


for i in range(
    len(search_passages)
):

    source_row = (
        search_passages.iloc[i]
    )

    for rank in range(
        1,
        TOP_K + 1
    ):

        target_index = int(
            indices[i][rank]
        )

        semantic_score = float(
            scores[i][rank]
        )

        if (
            semantic_score
            < SEMANTIC_THRESHOLD
        ):
            continue

        target_row = (
            search_passages.iloc[
                target_index
            ]
        )

        # Skip same document
        if (
            source_row["document_id"]
            ==
            target_row["document_id"]
        ):
            continue


        pair_key = tuple(
            sorted(
                [
                    str(
                        source_row[
                            "chunk_id"
                        ]
                    ),
                    str(
                        target_row[
                            "chunk_id"
                        ]
                    ),
                ]
            )
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(
            pair_key
        )


        lexical_score = (
            lexical_similarity(
                source_row[
                    "positive"
                ],
                target_row[
                    "positive"
                ],
            )
        )


        candidates.append(
            {
                "semantic_similarity":
                    round(
                        semantic_score,
                        4,
                    ),

                "lexical_similarity":
                    round(
                        lexical_score,
                        4,
                    ),

                "document_a":
                    source_row[
                        "document_id"
                    ],

                "document_b":
                    target_row[
                        "document_id"
                    ],

                "chunk_a":
                    source_row[
                        "chunk_id"
                    ],

                "chunk_b":
                    target_row[
                        "chunk_id"
                    ],

                "title_a":
                    source_row[
                        "title"
                    ],

                "title_b":
                    target_row[
                        "title"
                    ],

                "passage_a":
                    source_row[
                        "positive"
                    ],

                "passage_b":
                    target_row[
                        "positive"
                    ],

                # Fill this manually
                "confirmed_duplicate":
                    "",

                "review_note":
                    "",
            }
        )


# ==========================================================
# SAVE CANDIDATES
# ==========================================================

candidate_df = pd.DataFrame(
    candidates
)

if len(candidate_df):

    candidate_df = (
        candidate_df
        .sort_values(
            [
                "semantic_similarity",
                "lexical_similarity",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


candidate_df.to_csv(
    CANDIDATE_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 65)
print("DUPLICATE DETECTION COMPLETED")
print("=" * 65)

print(
    f"Unique passages       : "
    f"{len(passages):,}"
)

print(
    f"Exact duplicate rows  : "
    f"{len(exact_duplicates):,}"
)

print(
    f"Near-dup candidates   : "
    f"{len(candidate_df):,}"
)

print(
    f"\nExact report:\n"
    f"{EXACT_FILE}"
)

print(
    f"\nCandidate report:\n"
    f"{CANDIDATE_FILE}"
)

print("=" * 65)
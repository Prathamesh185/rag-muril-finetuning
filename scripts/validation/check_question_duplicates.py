import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = (
    "data/validation_v2/final_dataset_grouped.csv"
)

OUTPUT_DIR = Path(
    "data/validation_v2/question_checks"
)

EXACT_OUTPUT = (
    OUTPUT_DIR / "exact_duplicate_questions.csv"
)

NEAR_OUTPUT = (
    OUTPUT_DIR / "near_duplicate_questions.csv"
)

COPY_OUTPUT = (
    OUTPUT_DIR / "question_copying_candidates.csv"
)


# Near duplicate settings
TOP_K = 5

# 0.90 is intentionally strict.
# We want almost identical questions, not merely same topic.
NEAR_DUP_THRESHOLD = 0.90

# Question-passage copying threshold
COPY_SEQUENCE_THRESHOLD = 0.75

# Minimum question length for copying check
MIN_QUESTION_CHARS = 25


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = text.lower()

    # Remove invisible Unicode characters
    text = (
        text
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
    )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s\u0900-\u097f]",
        " ",
        text
    )

    # Normalize whitespace
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
print("Loading grouped dataset...")
print("=" * 65)

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig"
)

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
        f"Missing columns: {missing_columns}"
    )


print(
    f"Question-passage pairs : {len(df):,}"
)

print(
    f"Unique questions       : "
    f"{df['question'].nunique():,}"
)

print(
    f"Unique chunks          : "
    f"{df['chunk_id'].nunique():,}"
)

print(
    f"Unique documents       : "
    f"{df['document_id'].nunique():,}"
)


# ==========================================================
# NORMALIZE QUESTIONS
# ==========================================================

print("\nNormalizing questions...")

df["normalized_question"] = (
    df["question"]
    .apply(normalize_text)
)


# ==========================================================
# 1. EXACT / NORMALIZED DUPLICATE QUESTIONS
# ==========================================================

print(
    "\nChecking exact duplicate questions..."
)

duplicate_mask = (
    df.duplicated(
        subset=["normalized_question"],
        keep=False,
    )
)

exact_duplicates = (
    df[
        duplicate_mask
    ]
    .copy()
    .sort_values(
        "normalized_question"
    )
)


exact_duplicates.to_csv(
    EXACT_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


print(
    f"Exact duplicate rows : "
    f"{len(exact_duplicates):,}"
)

print(
    f"Duplicate question groups : "
    f"{exact_duplicates['normalized_question'].nunique():,}"
)


# ==========================================================
# CREATE UNIQUE QUESTION TABLE
# ==========================================================

# For semantic / lexical duplicate search,
# we only need one copy of an exact question.

unique_questions = (
    df
    .drop_duplicates(
        subset=["normalized_question"],
        keep="first",
    )
    .reset_index(drop=True)
)


print(
    f"\nQuestions after exact deduplication : "
    f"{len(unique_questions):,}"
)


# ==========================================================
# 2. NEAR-DUPLICATE QUESTION SEARCH
# ==========================================================

print(
    "\nBuilding TF-IDF question representations..."
)

questions = (
    unique_questions[
        "normalized_question"
    ]
    .fillna("")
    .tolist()
)


# Character n-grams work well for Hindi paraphrase /
# near-copy detection because they catch small wording changes.

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=1,
    max_features=150000,
)


question_vectors = (
    vectorizer.fit_transform(
        questions
    )
)


print(
    f"Vector matrix shape : "
    f"{question_vectors.shape}"
)


# ==========================================================
# NEAREST NEIGHBOR SEARCH
# ==========================================================

print(
    "\nSearching near-duplicate questions..."
)

neighbors = NearestNeighbors(
    n_neighbors=min(
        TOP_K + 1,
        len(unique_questions)
    ),
    metric="cosine",
    algorithm="brute",
    n_jobs=-1,
)

neighbors.fit(
    question_vectors
)


distances, indices = (
    neighbors.kneighbors(
        question_vectors
    )
)


near_candidates = []

seen_pairs = set()


for i in range(
    len(unique_questions)
):

    row_a = unique_questions.iloc[i]

    for neighbor_position in range(
        1,
        indices.shape[1]
    ):

        j = int(
            indices[i][neighbor_position]
        )

        if i == j:
            continue

        similarity = (
            1.0
            - float(
                distances[i][neighbor_position]
            )
        )


        if similarity < NEAR_DUP_THRESHOLD:
            continue


        row_b = unique_questions.iloc[j]


        # Avoid storing A-B and B-A separately
        pair_key = tuple(
            sorted(
                [
                    str(row_a["pair_id"]),
                    str(row_b["pair_id"]),
                ]
            )
        )

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(
            pair_key
        )


        # Extra character-level similarity
        sequence_similarity = (
            SequenceMatcher(
                None,
                row_a[
                    "normalized_question"
                ],
                row_b[
                    "normalized_question"
                ],
            )
            .ratio()
        )


        same_chunk = (
            str(row_a["chunk_id"])
            ==
            str(row_b["chunk_id"])
        )

        same_document = (
            str(row_a["document_id"])
            ==
            str(row_b["document_id"])
        )

        same_split_group = (
            str(row_a["split_group_id"])
            ==
            str(row_b["split_group_id"])
        )


        near_candidates.append(
            {
                "tfidf_similarity":
                    round(similarity, 4),

                "sequence_similarity":
                    round(
                        sequence_similarity,
                        4
                    ),

                "question_a":
                    row_a["question"],

                "question_b":
                    row_b["question"],

                "pair_id_a":
                    row_a["pair_id"],

                "pair_id_b":
                    row_b["pair_id"],

                "chunk_id_a":
                    row_a["chunk_id"],

                "chunk_id_b":
                    row_b["chunk_id"],

                "document_a":
                    row_a["document_id"],

                "document_b":
                    row_b["document_id"],

                "split_group_a":
                    row_a["split_group_id"],

                "split_group_b":
                    row_b["split_group_id"],

                "same_chunk":
                    same_chunk,

                "same_document":
                    same_document,

                "same_split_group":
                    same_split_group,

                # Fill manually later
                "review_decision":
                    "",

                "review_note":
                    "",
            }
        )


near_df = pd.DataFrame(
    near_candidates
)


if len(near_df):

    near_df = (
        near_df
        .sort_values(
            [
                "tfidf_similarity",
                "sequence_similarity",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


near_df.to_csv(
    NEAR_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


print(
    f"Near-duplicate candidates : "
    f"{len(near_df):,}"
)


# ==========================================================
# 3. QUESTION COPYING FROM POSITIVE PASSAGE
# ==========================================================

print(
    "\nChecking question-passage copying..."
)


def question_passage_copy_score(
    question,
    passage,
):

    q = normalize_text(
        question
    )

    p = normalize_text(
        passage
    )

    if not q or not p:

        return 0.0


    # If the entire normalized question occurs
    # directly inside the passage, this is a very
    # strong copying signal.

    if q in p:

        return 1.0


    # Compare question with sliding windows
    # from the passage.
    #
    # This catches cases where most of the question
    # was copied with only a few small changes.

    question_words = q.split()

    passage_words = p.split()


    if len(question_words) == 0:
        return 0.0


    window_size = len(
        question_words
    )


    # Allow slightly larger windows
    candidate_sizes = {
        max(1, window_size - 2),
        window_size,
        window_size + 2,
    }


    best_score = 0.0


    for size in candidate_sizes:

        if size > len(
            passage_words
        ):
            continue


        for start in range(
            0,
            len(passage_words) - size + 1
        ):

            window = " ".join(
                passage_words[
                    start:
                    start + size
                ]
            )


            score = SequenceMatcher(
                None,
                q,
                window,
            ).ratio()


            if score > best_score:

                best_score = score


    return best_score


copy_candidates = []


for row_index, row in df.iterrows():

    question = str(
        row["question"]
    )


    if len(question) < MIN_QUESTION_CHARS:
        continue


    copy_score = (
        question_passage_copy_score(
            row["question"],
            row["positive"],
        )
    )


    if (
        copy_score
        >= COPY_SEQUENCE_THRESHOLD
    ):

        copy_candidates.append(
            {
                "copy_score":
                    round(
                        copy_score,
                        4
                    ),

                "question":
                    row["question"],

                "positive":
                    row["positive"],

                "pair_id":
                    row["pair_id"],

                "chunk_id":
                    row["chunk_id"],

                "document_id":
                    row["document_id"],

                "split_group_id":
                    row["split_group_id"],

                # Fill manually
                "review_decision":
                    "",

                "review_note":
                    "",
            }
        )


copy_df = pd.DataFrame(
    copy_candidates
)


if len(copy_df):

    copy_df = (
        copy_df
        .sort_values(
            "copy_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


copy_df.to_csv(
    COPY_OUTPUT,
    index=False,
    encoding="utf-8-sig",
)


print(
    f"Question-copy candidates : "
    f"{len(copy_df):,}"
)


# ==========================================================
# SUMMARY
# ==========================================================

print(
    "\n"
    + "=" * 65
)

print(
    "QUESTION VALIDATION COMPLETE"
)

print(
    "=" * 65
)


print(
    f"Total question-passage pairs  : "
    f"{len(df):,}"
)

print(
    f"Exact duplicate rows          : "
    f"{len(exact_duplicates):,}"
)

print(
    f"Exact duplicate groups        : "
    f"{exact_duplicates['normalized_question'].nunique():,}"
)

print(
    f"Near-duplicate candidates     : "
    f"{len(near_df):,}"
)

print(
    f"Question-copy candidates      : "
    f"{len(copy_df):,}"
)


print(
    "\nReports saved to:"
)

print(
    f"1. {EXACT_OUTPUT}"
)

print(
    f"2. {NEAR_OUTPUT}"
)

print(
    f"3. {COPY_OUTPUT}"
)


print(
    "\nIMPORTANT:"
)

print(
    "Do NOT delete near-duplicate or copying "
    "candidates automatically."
)

print(
    "Review them first."
)

print(
    "\nNext step after review:"
)

print(
    "Create the cleaned final_dataset_v2.csv, "
    "then perform the final group-aware "
    "train/valid/test split."
)

print(
    "=" * 65
)
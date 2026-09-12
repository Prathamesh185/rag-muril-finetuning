import os
import re
import unicodedata

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

TRAIN_FILE = "data/training_v2/train.csv"
MODEL_PATH = "models/fine_tuned_muril_v2"

OUTPUT_FILE = "data/training_v2/train_hard_negatives_v2.csv"
REVIEW_FILE = "data/training_v2/hard_negatives_review_v2.csv"

TOP_K = 100
HARD_NEGATIVES_PER_QUERY = 2
BATCH_SIZE = 64

# Avoid negatives that are almost as similar as the positive
RELATIVE_MARGIN = 0.05

# Avoid negatives that are too easy
MAX_SCORE_MARGIN = 0.30

# Avoid very similar / near-duplicate passages
MAX_TOKEN_JACCARD = 0.85

# Low positive similarity may indicate noisy QA pairs
MIN_POSITIVE_SCORE = 0.45
SKIP_LOW_POSITIVE = True

REVIEW_SAMPLE_SIZE = 200
RANDOM_SEED = 42


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(text):
    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_tokens(text):
    text = normalize_text(text)

    return set(
        re.findall(
            r"\w+",
            text,
            flags=re.UNICODE
        )
    )


def token_jaccard(text_a, text_b):
    tokens_a = get_tokens(text_a)
    tokens_b = get_tokens(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    return (
        len(tokens_a & tokens_b)
        /
        len(tokens_a | tokens_b)
    )


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("=" * 60)
print("HARD NEGATIVE MINING - V2")
print("=" * 60)

print("\nLoading training data...")

df = pd.read_csv(
    TRAIN_FILE,
    encoding="utf-8-sig",
    dtype={
        "pair_id": str,
        "chunk_id": str,
        "document_id": str,
    }
)


required_columns = [
    "question",
    "positive",
    "pair_id",
    "chunk_id",
    "document_id",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = (
    df
    .dropna(
        subset=required_columns
    )
    .reset_index(drop=True)
)


print(f"Training pairs: {len(df):,}")
print(f"Unique chunks: {df['chunk_id'].nunique():,}")
print(f"Unique documents: {df['document_id'].nunique():,}")


# =========================================================
# OPTIONAL GROUP COLUMN
# =========================================================

possible_group_columns = [
    "split_group_id",
    "duplicate_group_id",
    "dup_group_id",
    "group_id",
]

group_column = None

for col in possible_group_columns:
    if col in df.columns:
        group_column = col
        break


if group_column:
    df[group_column] = (
        df[group_column]
        .fillna("")
        .astype(str)
    )

    print(
        f"Using group column: {group_column}"
    )
else:
    print(
        "No duplicate/split group column found."
    )


# =========================================================
# NORMALIZED QUESTIONS
# =========================================================

df["_normalized_question"] = (
    df["question"]
    .map(normalize_text)
)


# =========================================================
# PROTECT MULTIPLE POSITIVES FOR SAME QUESTION
# =========================================================

question_to_positive_chunks = (
    df
    .groupby("_normalized_question")["chunk_id"]
    .apply(set)
    .to_dict()
)


# =========================================================
# BUILD UNIQUE TRAINING CORPUS
# =========================================================

corpus_columns = [
    "chunk_id",
    "document_id",
    "positive",
]

if group_column:
    corpus_columns.append(
        group_column
    )


passage_df = (
    df[corpus_columns]
    .drop_duplicates(
        subset=["chunk_id"]
    )
    .reset_index(drop=True)
)


passage_df["_normalized_text"] = (
    passage_df["positive"]
    .map(normalize_text)
)


print(
    f"Mining corpus passages: {len(passage_df):,}"
)


# =========================================================
# LOAD MODEL
# =========================================================

print(
    "\nLoading Fine-Tuned MuRIL V2..."
)

model = SentenceTransformer(
    MODEL_PATH
)


# =========================================================
# ENCODE PASSAGES
# =========================================================

print(
    "\nEncoding passages..."
)

passage_embeddings = model.encode(
    passage_df["positive"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

passage_embeddings = np.asarray(
    passage_embeddings,
    dtype=np.float32
)


# =========================================================
# BUILD FAISS INDEX
# =========================================================

dimension = passage_embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    passage_embeddings
)


print(
    f"FAISS passages indexed: {index.ntotal:,}"
)


# =========================================================
# ENCODE QUESTIONS
# =========================================================

print(
    "\nEncoding questions..."
)

question_embeddings = model.encode(
    df["question"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

question_embeddings = np.asarray(
    question_embeddings,
    dtype=np.float32
)


# =========================================================
# ENCODE KNOWN POSITIVES
# =========================================================

print(
    "\nEncoding positive passages..."
)

positive_embeddings = model.encode(
    df["positive"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
)

positive_embeddings = np.asarray(
    positive_embeddings,
    dtype=np.float32
)


positive_scores = np.sum(
    question_embeddings
    * positive_embeddings,
    axis=1
)


# =========================================================
# SEARCH
# =========================================================

effective_top_k = min(
    TOP_K,
    len(passage_df)
)


print(
    f"\nSearching Top-{effective_top_k} candidates..."
)


search_scores, search_indices = index.search(
    question_embeddings,
    effective_top_k
)


# =========================================================
# MINE HARD NEGATIVES
# =========================================================

print(
    "\nMining hard negatives..."
)


rows = []

filter_counts = {
    "same_positive": 0,
    "known_positive": 0,
    "same_document": 0,
    "same_group": 0,
    "exact_duplicate": 0,
    "near_duplicate": 0,
    "too_close": 0,
    "too_easy": 0,
    "low_positive": 0,
}


full_count = 0
partial_count = 0
zero_count = 0
low_positive_queries = 0


for query_idx, row in df.iterrows():

    question = str(
        row["question"]
    )

    positive_text = str(
        row["positive"]
    )

    pair_id = str(
        row["pair_id"]
    )

    positive_chunk_id = str(
        row["chunk_id"]
    )

    positive_document_id = str(
        row["document_id"]
    )

    normalized_question = (
        row["_normalized_question"]
    )

    normalized_positive = normalize_text(
        positive_text
    )

    positive_score = float(
        positive_scores[query_idx]
    )


    # -----------------------------------------------------
    # LOW POSITIVE SCORE
    # -----------------------------------------------------

    if positive_score < MIN_POSITIVE_SCORE:

        low_positive_queries += 1

        if SKIP_LOW_POSITIVE:
            filter_counts[
                "low_positive"
            ] += 1

            continue


    known_positive_chunks = (
        question_to_positive_chunks.get(
            normalized_question,
            {positive_chunk_id},
        )
    )


    if group_column:
        positive_group_id = str(
            row[group_column]
        )
    else:
        positive_group_id = None


    selected = 0

    selected_negative_ids = set()


    # =====================================================
    # LOOP THROUGH RETRIEVED CANDIDATES
    # =====================================================

    for rank, (
        candidate_score,
        corpus_idx
    ) in enumerate(
        zip(
            search_scores[query_idx],
            search_indices[query_idx]
        ),
        start=1,
    ):

        if corpus_idx < 0:
            continue


        candidate = passage_df.iloc[
            int(corpus_idx)
        ]


        negative_chunk_id = str(
            candidate["chunk_id"]
        )

        negative_document_id = str(
            candidate["document_id"]
        )

        negative_text = str(
            candidate["positive"]
        )

        normalized_negative = str(
            candidate["_normalized_text"]
        )

        candidate_score = float(
            candidate_score
        )


        # -------------------------------------------------
        # FILTER 1: assigned positive
        # -------------------------------------------------

        if (
            negative_chunk_id
            == positive_chunk_id
        ):
            filter_counts[
                "same_positive"
            ] += 1

            continue


        # -------------------------------------------------
        # FILTER 2: any known positive for same question
        # -------------------------------------------------

        if (
            negative_chunk_id
            in known_positive_chunks
        ):
            filter_counts[
                "known_positive"
            ] += 1

            continue


        # -------------------------------------------------
        # FILTER 3: same document
        # -------------------------------------------------

        if (
            negative_document_id
            == positive_document_id
        ):
            filter_counts[
                "same_document"
            ] += 1

            continue


        # -------------------------------------------------
        # FILTER 4: same duplicate/split group
        # -------------------------------------------------

        negative_group_id = None

        if group_column:

            negative_group_id = str(
                candidate[group_column]
            )

            if (
                positive_group_id
                and
                negative_group_id
                and
                positive_group_id
                == negative_group_id
            ):
                filter_counts[
                    "same_group"
                ] += 1

                continue


        # -------------------------------------------------
        # FILTER 5: exact duplicate text
        # -------------------------------------------------

        if (
            normalized_negative
            == normalized_positive
        ):
            filter_counts[
                "exact_duplicate"
            ] += 1

            continue


        # -------------------------------------------------
        # FILTER 6: near-duplicate text
        # -------------------------------------------------

        lexical_overlap = token_jaccard(
            positive_text,
            negative_text
        )


        if (
            lexical_overlap
            >= MAX_TOKEN_JACCARD
        ):
            filter_counts[
                "near_duplicate"
            ] += 1

            continue


        # -------------------------------------------------
        # FILTER 7: too close to positive
        # -------------------------------------------------

        max_negative_score = (
            positive_score
            * (1.0 - RELATIVE_MARGIN)
        )


        if (
            candidate_score
            > max_negative_score
        ):
            filter_counts[
                "too_close"
            ] += 1

            continue


        # -------------------------------------------------
        # SCORE MARGIN
        # -------------------------------------------------

        score_margin = (
            positive_score
            - candidate_score
        )


        # -------------------------------------------------
        # FILTER 8: too easy
        # -------------------------------------------------

        if (
            score_margin
            > MAX_SCORE_MARGIN
        ):
            filter_counts[
                "too_easy"
            ] += 1

            continue


        # -------------------------------------------------
        # DUPLICATE SELECTED NEGATIVE
        # -------------------------------------------------

        if (
            negative_chunk_id
            in selected_negative_ids
        ):
            continue


        selected_negative_ids.add(
            negative_chunk_id
        )


        # -------------------------------------------------
        # NORMALIZED MARGIN
        # -------------------------------------------------

        normalized_margin = (
            score_margin
            /
            max(
                abs(positive_score),
                1e-8
            )
        )


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        output_row = {

            "pair_id":
                pair_id,

            "question":
                question,

            "positive":
                positive_text,

            "negative":
                negative_text,

            "positive_chunk_id":
                positive_chunk_id,

            "negative_chunk_id":
                negative_chunk_id,

            "positive_document_id":
                positive_document_id,

            "negative_document_id":
                negative_document_id,

            "positive_score":
                positive_score,

            "negative_score":
                candidate_score,

            "score_margin":
                score_margin,

            "normalized_margin":
                normalized_margin,

            "negative_rank":
                rank,

            "token_jaccard":
                lexical_overlap,
        }


        if group_column:

            output_row[
                "positive_group_id"
            ] = positive_group_id

            output_row[
                "negative_group_id"
            ] = negative_group_id


        rows.append(
            output_row
        )


        selected += 1


        if (
            selected
            >= HARD_NEGATIVES_PER_QUERY
        ):
            break


    # =====================================================
    # COVERAGE STATS
    # =====================================================

    if (
        selected
        == HARD_NEGATIVES_PER_QUERY
    ):
        full_count += 1

    elif selected > 0:
        partial_count += 1

    else:
        zero_count += 1


    if (
        (query_idx + 1) % 2000
        == 0
    ):
        print(
            f"Processed "
            f"{query_idx + 1:,} / "
            f"{len(df):,}"
        )


# =========================================================
# CREATE OUTPUT DATAFRAME
# =========================================================

hard_df = pd.DataFrame(
    rows
)


if hard_df.empty:
    raise RuntimeError(
        "No hard negatives were mined."
    )


# =========================================================
# FINAL DEDUP
# =========================================================

before_dedup = len(
    hard_df
)

hard_df = (
    hard_df
    .drop_duplicates(
        subset=[
            "pair_id",
            "negative_chunk_id",
        ]
    )
    .reset_index(drop=True)
)

after_dedup = len(
    hard_df
)


# =========================================================
# SORT
# =========================================================

hard_df = (
    hard_df
    .sort_values(
        by=[
            "pair_id",
            "negative_rank",
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# SAVE FULL DATASET
# =========================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
    exist_ok=True
)


hard_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# CREATE REVIEW SAMPLE
# =========================================================

review_size = min(
    REVIEW_SAMPLE_SIZE,
    len(hard_df)
)


review_df = hard_df.sample(
    n=review_size,
    random_state=RANDOM_SEED
).copy()


review_df[
    "review_label"
] = ""

review_df[
    "review_notes"
] = ""


review_df.to_csv(
    REVIEW_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# FINAL REPORT
# =========================================================

print()
print("=" * 60)
print("HARD NEGATIVE MINING COMPLETE")
print("=" * 60)


print(
    f"\nTraining queries: "
    f"{len(df):,}"
)

print(
    f"Hard-negative triplets: "
    f"{len(hard_df):,}"
)

print(
    f"Unique queries with negatives: "
    f"{hard_df['pair_id'].nunique():,}"
)


print(
    f"\nQueries with 2 negatives: "
    f"{full_count:,}"
)

print(
    f"Queries with 1 negative: "
    f"{partial_count:,}"
)

print(
    f"Queries with 0 negatives: "
    f"{zero_count:,}"
)


print(
    f"\nLow-positive-score queries "
    f"(< {MIN_POSITIVE_SCORE}): "
    f"{low_positive_queries:,}"
)


# =========================================================
# NEGATIVES PER QUERY
# =========================================================

print(
    "\nNegatives per query:"
)

print(
    hard_df
    .groupby("pair_id")
    .size()
    .value_counts()
    .sort_index()
)


# =========================================================
# SCORE STATS
# =========================================================

print(
    "\nPositive score statistics:"
)

print(
    hard_df[
        "positive_score"
    ]
    .describe()
)


print(
    "\nNegative score statistics:"
)

print(
    hard_df[
        "negative_score"
    ]
    .describe()
)


print(
    "\nScore margin statistics:"
)

print(
    hard_df[
        "score_margin"
    ]
    .describe()
)


print(
    "\nNormalized margin statistics:"
)

print(
    hard_df[
        "normalized_margin"
    ]
    .describe()
)


print(
    "\nNegative rank statistics:"
)

print(
    hard_df[
        "negative_rank"
    ]
    .describe()
)


# =========================================================
# FILTER STATS
# =========================================================

print(
    "\nCandidates rejected:"
)

for name, count in filter_counts.items():

    print(
        f"{name}: {count:,}"
    )


print(
    f"\nDuplicate triplets removed: "
    f"{before_dedup - after_dedup:,}"
)


# =========================================================
# HARDEST EXAMPLES
# =========================================================

print(
    "\n10 hardest accepted negatives:"
)


print(
    hard_df[
        [
            "question",
            "positive_score",
            "negative_score",
            "score_margin",
            "negative_rank",
        ]
    ]
    .sort_values(
        "score_margin"
    )
    .head(10)
    .to_string(
        index=False
    )
)


# =========================================================
# OUTPUT FILES
# =========================================================

print(
    f"\nFull dataset saved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nReview sample saved to:\n"
    f"{REVIEW_FILE}"
)

print(
    "\nNext step:"
)

print(
    "Review the 200 sampled negatives before training V3."
)
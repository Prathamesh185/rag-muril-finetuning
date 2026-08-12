import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = (
    "data/validation_v2/final_dataset_grouped.csv"
)

OUTPUT_DIR = Path(
    "data/validation_v2/question_checks"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "question_copying_candidates_fast.csv"
)

# Fast screening thresholds
MIN_QUESTION_CHARS = 25

# Percentage of question words also present in passage
WORD_OVERLAP_THRESHOLD = 0.90

# SequenceMatcher runs only after fast filtering
SEQUENCE_THRESHOLD = 0.85


# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = text.lower()

    text = (
        text
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
    )

    text = re.sub(
        r"[^\w\s\u0900-\u097f]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# WORD OVERLAP
# ==========================================================

def word_overlap_score(question, passage):

    q = normalize_text(question)
    p = normalize_text(passage)

    if not q or not p:
        return 0.0

    q_words = q.split()
    p_words = set(
        p.split()
    )

    if not q_words:
        return 0.0

    matched = sum(
        1
        for word in q_words
        if word in p_words
    )

    return (
        matched
        / len(q_words)
    )


# ==========================================================
# LOCAL SEQUENCE SIMILARITY
# ==========================================================

def local_sequence_similarity(
    question,
    passage,
):

    q = normalize_text(question)
    p = normalize_text(passage)

    if not q or not p:
        return 0.0


    # Exact occurrence
    if q in p:
        return 1.0


    q_words = q.split()
    p_words = p.split()


    if not q_words:
        return 0.0


    q_len = len(
        q_words
    )


    # Compare only windows close to question length
    candidate_sizes = [
        max(1, q_len - 2),
        q_len,
        q_len + 2,
    ]


    best_score = 0.0


    for size in candidate_sizes:

        if size > len(p_words):
            continue


        for start in range(
            0,
            len(p_words) - size + 1
        ):

            window = " ".join(
                p_words[
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


            # Early exit if already almost exact
            if best_score >= 0.98:
                return best_score


    return best_score


# ==========================================================
# LOAD DATA
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
        f"Missing required columns: "
        f"{missing_columns}"
    )


print(
    f"Total pairs : {len(df):,}"
)


# ==========================================================
# FAST SCREENING
# ==========================================================

print(
    "\nRunning fast word-overlap screening..."
)


fast_candidates = []


for index, row in df.iterrows():

    question = str(row["question"])
    passage = str(row["positive"])

    if len(question) < MIN_QUESTION_CHARS:
        continue

    q = normalize_text(question)
    p = normalize_text(passage)

    # ------------------------------------------------------
    # STRONG COPYING CASE:
    # entire question appears directly inside passage
    # ------------------------------------------------------

    exact_copy = q in p

    # ------------------------------------------------------
    # FAST WORD OVERLAP
    # ------------------------------------------------------

    overlap = word_overlap_score(
        question,
        passage,
    )

    # Keep if:
    # 1. Exact question appears in passage
    # OR
    # 2. Very high word overlap
    if not exact_copy and overlap < WORD_OVERLAP_THRESHOLD:
        continue

    fast_candidates.append(
    {
        "row_index": index,

        "exact_copy":
            exact_copy,

        "word_overlap":
            overlap,

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
    }
    )


print(
    f"Fast overlap candidates : "
    f"{len(fast_candidates):,}"
)


# ==========================================================
# ACCURATE SECOND-STAGE CHECK
# ==========================================================

print(
    "\nRunning detailed comparison "
    "only on suspicious rows..."
)


final_candidates = []


for i, candidate in enumerate(
    fast_candidates,
    start=1,
):

    sequence_score = (
        local_sequence_similarity(
            candidate["question"],
            candidate["positive"],
        )
    )


    if sequence_score < SEQUENCE_THRESHOLD:
        continue


    final_candidates.append(
        {
            "word_overlap":
                round(
                    candidate[
                        "word_overlap"
                    ],
                    4,
                ),

            "sequence_similarity":
                round(
                    sequence_score,
                    4,
                ),

            "question":
                candidate[
                    "question"
                ],

            "positive":
                candidate[
                    "positive"
                ],

            "pair_id":
                candidate[
                    "pair_id"
                ],

            "chunk_id":
                candidate[
                    "chunk_id"
                ],

            "document_id":
                candidate[
                    "document_id"
                ],

            "split_group_id":
                candidate[
                    "split_group_id"
                ],

            # Fill after review
            "review_decision":
                "",

            "review_note":
                "",
        }
    )


    if i % 500 == 0:

        print(
            f"Checked "
            f"{i:,}/"
            f"{len(fast_candidates):,}"
        )


# ==========================================================
# CREATE OUTPUT DATAFRAME
# ==========================================================

result_df = pd.DataFrame(
    final_candidates
)


if len(result_df):

    result_df = (
        result_df
        .sort_values(
            [
                "sequence_similarity",
                "word_overlap",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ==========================================================
# SAVE
# ==========================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# SUMMARY
# ==========================================================

print(
    "\n"
    + "=" * 65
)

print(
    "QUESTION COPYING CHECK COMPLETE"
)

print(
    "=" * 65
)


print(
    f"Total pairs                 : "
    f"{len(df):,}"
)

print(
    f"Fast overlap candidates     : "
    f"{len(fast_candidates):,}"
)

print(
    f"Final copying candidates    : "
    f"{len(result_df):,}"
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\nReview decisions to use:"
)

print(
    "COPYING_PROBLEM"
)

print(
    "KEEP"
)

print(
    "UNCERTAIN"
)

print(
    "=" * 65
)
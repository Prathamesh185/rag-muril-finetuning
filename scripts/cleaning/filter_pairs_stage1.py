import re
from pathlib import Path

import pandas as pd

# =====================================================
# Paths
# =====================================================

INPUT_FILE = Path("data/training/generated_pairs.csv")

OUTPUT_DIR = Path("data/cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_FILE = OUTPUT_DIR / "keep_stage1.csv"
REMOVE_FILE = OUTPUT_DIR / "remove_stage1.csv"

# =====================================================
# Thresholds
# =====================================================

MIN_QUESTION_WORDS = 4
MIN_POSITIVE_WORDS = 15
MAX_QUESTION_WORDS = 40

OCR_CHARS = {"�", "□", "■", "◆", "◼", "◻"}

# =====================================================
# Helpers
# =====================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_question(text):
    text = clean_text(text).lower()

    text = re.sub(r"[^\w\u0900-\u097f\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def repeated_word(text):
    words = text.split()

    if len(words) < 4:
        return False

    return len(set(words)) == 1


def has_hindi(text):
    return re.search(r"[\u0900-\u097F]", text) is not None


def has_ocr(text):
    return any(ch in text for ch in OCR_CHARS)


# =====================================================
# Load
# =====================================================

df = pd.read_csv(INPUT_FILE)

keep_rows = []
remove_rows = []

seen_questions = set()
seen_pairs = set()

# =====================================================
# Cleaning
# =====================================================

for _, row in df.iterrows():

    question = clean_text(row["question"])
    positive = clean_text(row["positive"])

    reason = None

    # ---------------------------
    # Empty
    # ---------------------------

    if question == "":
        reason = "empty_question"

    elif positive == "":
        reason = "empty_positive"

    # ---------------------------
    # Length
    # ---------------------------

    elif len(question.split()) < MIN_QUESTION_WORDS:
        reason = "short_question"

    elif len(question.split()) > MAX_QUESTION_WORDS:
        reason = "long_question"

    elif len(positive.split()) < MIN_POSITIVE_WORDS:
        reason = "short_positive"

    # ---------------------------
    # Hindi
    # ---------------------------

    elif not has_hindi(question):
        reason = "no_hindi"

    # ---------------------------
    # Question format
    # ---------------------------

    elif not question.endswith("?"):
        reason = "missing_question_mark"

    elif question.count("?") > 2:
        reason = "multiple_question_marks"

    # ---------------------------
    # OCR
    # ---------------------------

    elif has_ocr(question) or has_ocr(positive):
        reason = "ocr_garbage"

    # ---------------------------
    # Same question & answer
    # ---------------------------

    elif normalize_question(question) == normalize_question(positive):
        reason = "question_equals_positive"

    # ---------------------------
    # Repeated word
    # ---------------------------

    elif repeated_word(question):
        reason = "repeated_word"

    # ---------------------------
    # Duplicate question
    # ---------------------------

    norm_question = normalize_question(question)

    if reason is None:

        if norm_question in seen_questions:
            reason = "duplicate_question"

        else:
            seen_questions.add(norm_question)

    # ---------------------------
    # Duplicate pair
    # ---------------------------

    norm_pair = (
        normalize_question(question),
        normalize_question(positive),
    )

    if reason is None:

        if norm_pair in seen_pairs:
            reason = "duplicate_pair"

        else:
            seen_pairs.add(norm_pair)

    # ---------------------------
    # Save
    # ---------------------------

    if reason is None:

        row["question"] = question
        row["positive"] = positive

        keep_rows.append(row)

    else:

        row["question"] = question
        row["positive"] = positive
        row["remove_reason"] = reason

        remove_rows.append(row)

# =====================================================
# Save
# =====================================================

keep_df = pd.DataFrame(keep_rows)
remove_df = pd.DataFrame(remove_rows)

keep_df.to_csv(KEEP_FILE, index=False, encoding="utf-8-sig")
remove_df.to_csv(REMOVE_FILE, index=False, encoding="utf-8-sig")

# =====================================================
# Summary
# =====================================================

print("=" * 60)
print(f"Original pairs : {len(df)}")
print(f"Kept           : {len(keep_df)}")
print(f"Removed        : {len(remove_df)}")
print("=" * 60)

if len(remove_df):
    print("\nRemoval reasons:\n")
    print(remove_df["remove_reason"].value_counts())

print("\nSaved:")
print(KEEP_FILE)
print(REMOVE_FILE)
print("=" * 60)
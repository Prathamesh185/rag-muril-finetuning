import re
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/chunks/selected_chunks.csv")
OUTPUT_FILE = Path("data/chunks/selected_chunks_fixed.csv")

MAX_WORDS = 200
OVERLAP_SENTENCES = 1
LARGE_CHUNK_THRESHOLD = 400


# ---------------------------------------------------
# Sentence Splitter
# ---------------------------------------------------

def split_sentences(text):
    """
    Improved sentence splitter.
    Handles Hindi punctuation and common separators.
    """

    text = str(text)

    text = text.replace("\n", " ")

    # Split after Hindi/English sentence endings
    text = re.sub(r"([।!?])\s+", r"\1\n", text)

    # Split after | used as sentence delimiter
    text = re.sub(r"\|\s*", "|\n", text)

    # Split after colon followed by a capital/Devanagari word
    text = re.sub(r":\s+", ":\n", text)

    sentences = []

    for line in text.split("\n"):

        line = line.strip()

        if len(line) > 5:
            sentences.append(line)

    return sentences


# ---------------------------------------------------
# Chunk Creator
# ---------------------------------------------------

def create_chunks(sentences):

    chunks = []

    i = 0

    while i < len(sentences):

        current = []
        words = 0

        j = i

        while j < len(sentences):

            sentence = sentences[j]

            wc = len(sentence.split())

            if current and words + wc > MAX_WORDS:
                break

            current.append(sentence)

            words += wc

            j += 1

        chunk = " ".join(current).strip()

        if len(chunk.split()) >= 40:
            chunks.append(chunk)

        i = max(j - OVERLAP_SENTENCES, i + 1)

    return chunks


# ---------------------------------------------------
# Main
# ---------------------------------------------------

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

fixed_rows = []

replaced = 0

for _, row in df.iterrows():

    if row["word_count"] <= LARGE_CHUNK_THRESHOLD:

        fixed_rows.append(row.to_dict())

        continue

    replaced += 1

    sentences = split_sentences(row["chunk_text"])

    new_chunks = create_chunks(sentences)

    for idx, chunk in enumerate(new_chunks):

        new_row = row.to_dict()

        new_row["chunk_text"] = chunk
        new_row["word_count"] = len(chunk.split())
        new_row["char_count"] = len(chunk)

        # preserve original id with subchunk suffix
        new_row["chunk_id"] = f"{row['chunk_id']}_fix{idx}"

        fixed_rows.append(new_row)

fixed_df = pd.DataFrame(fixed_rows)

fixed_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("Finished")
print("=" * 60)
print(f"Original rows      : {len(df)}")
print(f"Large chunks fixed : {replaced}")
print(f"Final rows         : {len(fixed_df)}")
print(f"Saved to           : {OUTPUT_FILE}")
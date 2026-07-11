import json
import re
import pandas as pd
from pathlib import Path

# Paths
SCRAPED_DIR = Path("data/scraped/hindi")
OUTPUT_FILE = Path("data/chunks/vikaspedia_passages.csv")


def clean_text(text):
    """Clean scraped text."""

    if not text:
        return ""

    # Remove invisible Unicode characters
    text = text.replace("\u200c", "")
    text = text.replace("\u200d", "")
    text = text.replace("\ufeff", "")

    # Remove unwanted navigation text
    remove_patterns = [
    "मुख्य पृष्ठ",
    "होम",
    "होम पेज",
    "साझा करें",
    "प्रिंट",
    "ईमेल",
    "फेसबुक",
    "ट्विटर",
    "अधिक जानकारी",
    "पिछला",
    "अगला",
    "Back",
    "Next"
    ]

    for pattern in remove_patterns:
        text = text.replace(pattern, " ")

    # Replace newlines with spaces
    text = text.replace("\n", " ")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove repeated punctuation
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"।{2,}", "।", text)

    text = re.sub(
        r"\|\s*Vikaspedia\s*-\s*Agriculture",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove source text
    text = re.sub(r"स्त्रोत\s*[:：][^\n।]*", "", text)
    text = re.sub(r"स्रोत\s*[:：][^\n।]*", "", text)

    # Remove author information
    text = re.sub(r"लेखन\s*[:：].*", "", text)
    text = re.sub(r"लेखक\s*[:：].*", "", text)
    
    text = re.sub(
        r"इस\s+(भाग|पृष्ठ|लेख)\s+में.*?जानकारी\s+दी\s+(गई|गयी|है)\।?",
        "",
        text
    )

    return text.strip()


def split_sentences(text):
    """Split text into sentences."""

    text = text.replace("\n", " ")

    sentences = re.split(r'(?<=[।.!?])\s+', text)

    clean_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence) > 20:
            clean_sentences.append(sentence)

    return clean_sentences


def create_chunks(sentences, max_words=220, overlap_sentences=3):
    """Create overlapping chunks."""

    chunks = []

    i = 0

    while i < len(sentences):

        current_chunk = []
        current_word_count = 0
        j = i

        while j < len(sentences):

            sentence = sentences[j]
            sentence_word_count = len(sentence.split())

            if current_chunk and current_word_count + sentence_word_count > max_words:
                break

            current_chunk.append(sentence)
            current_word_count += sentence_word_count
            j += 1

        chunk_text = " ".join(current_chunk)

        if len(chunk_text) > 80:
            chunks.append(chunk_text)

        # Keep overlap between consecutive chunks
        i = max(j - overlap_sentences, i + 1)

    return chunks


# ------------------------------------------------------------------
# Process all JSON files
# ------------------------------------------------------------------

all_chunks = []

# Sort files so output is always the same
files = sorted(SCRAPED_DIR.glob("*.json"))

print(f"Processing {len(files)} files...")

for index, file in enumerate(files):

    try:

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = clean_text(data.get("title", ""))
        summary = clean_text(data.get("summary", ""))
        content = clean_text(data.get("content", ""))
        url = data.get("url", "")

        # Combine title, summary and content
        full_text = ""

        if summary:
            full_text += summary + "\n"

        if content:
            full_text += content

        full_text = clean_text(full_text)

        if len(full_text) < 100:
            continue

        sentences = split_sentences(full_text)
        sentences = [s for s in sentences if s.strip()]

        chunks = create_chunks(
            sentences,
            max_words=200,
            overlap_sentences=1
        )

        document_id = file.stem

        for chunk_index, chunk in enumerate(chunks):

            row = {
                "chunk_id": f"{document_id}_chunk_{chunk_index}",
                "document_id": document_id,
                "chunk_index": chunk_index,
                "title": title,
                "chunk_text": chunk,
                "word_count": len(chunk.split()),
                "char_count": len(chunk),
                "language": "hi",
                "domain": "agriculture",
                "url": url,
                "source": "vikaspedia_hindi"
            }

            all_chunks.append(row)

        if (index + 1) % 200 == 0:
            print(f"{index + 1}/{len(files)} files processed")

    except Exception as e:
        print(f"Error processing {file.name}: {e}")


# ------------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------------

df = pd.DataFrame(all_chunks)

# Normalize whitespace
df["chunk_text"] = df["chunk_text"].str.replace(r"\s+", " ", regex=True)
df["chunk_text"] = df["chunk_text"].str.strip()

# Update counts
df["word_count"] = df["chunk_text"].apply(lambda x: len(x.split()))
df["char_count"] = df["chunk_text"].apply(len)

# Remove duplicate chunks
df = df.drop_duplicates(subset=["chunk_text"])

# Keep only useful chunks
df = df[df["word_count"] >= 40]

# Save CSV
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n✅ Total passages:", len(df))
print("Saved to:", OUTPUT_FILE)

print(df[[
    "chunk_id",
    "chunk_index",
    "title",
    "word_count",
    "char_count",
    "chunk_text"
]].head(3))
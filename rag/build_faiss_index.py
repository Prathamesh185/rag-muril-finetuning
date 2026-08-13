import faiss
import pandas as pd
from pathlib import Path

from rag.config import encoder


# ==========================================================
# CONFIG
# ==========================================================

CORPUS_FILE = "data/chunks/vikaspedia_passages.csv"

INDEX_DIR = Path("data/index")

INDEX_FILE = INDEX_DIR / "finetuned.faiss"

METADATA_FILE = INDEX_DIR / "finetuned_metadata.csv"

BATCH_SIZE = 32


# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# LOAD CORPUS
# ==========================================================

print("=" * 60)
print("Loading Vikaspedia passages...")
print("=" * 60)

df = pd.read_csv(
    CORPUS_FILE,
    encoding="utf-8-sig"
)

print(f"Total rows          : {len(df):,}")
print(f"Unique chunk IDs    : {df['chunk_id'].nunique():,}")
print(f"Duplicate chunk IDs : {df['chunk_id'].duplicated().sum():,}")
print(f"Duplicate texts     : {df['chunk_text'].duplicated().sum():,}")


# ==========================================================
# BASIC CLEANING
# ==========================================================

df = df.dropna(
    subset=["chunk_id", "chunk_text"]
).copy()

df["chunk_text"] = (
    df["chunk_text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Remove empty passages
df = df[
    df["chunk_text"].str.len() > 0
].copy()

# Keep only one row per chunk
df = df.drop_duplicates(
    subset=["chunk_id"],
    keep="first"
)

# Remove exact duplicate passage text
df = df.drop_duplicates(
    subset=["chunk_text"],
    keep="first"
)

df = df.reset_index(drop=True)

print("\nAfter cleaning")
print("-" * 60)
print(f"Final passages : {len(df):,}")


# ==========================================================
# CREATE EMBEDDINGS
# ==========================================================

print("\n" + "=" * 60)
print("Encoding passages using Fine-Tuned MuRIL V2...")
print("=" * 60)

embeddings = encoder.encode(
    df["chunk_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

embeddings = embeddings.astype("float32")

print("\nEmbedding shape:", embeddings.shape)


# ==========================================================
# VERIFY EMBEDDINGS
# ==========================================================

if len(df) != embeddings.shape[0]:
    raise ValueError(
        "Number of passages and embeddings do not match."
    )

dimension = embeddings.shape[1]

print("Embedding dimension:", dimension)


# ==========================================================
# BUILD FAISS INDEX
# ==========================================================

print("\n" + "=" * 60)
print("Building FAISS Index...")
print("=" * 60)

# Because embeddings are normalized,
# Inner Product behaves like Cosine Similarity
index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print(f"Passages indexed : {index.ntotal:,}")


# ==========================================================
# SAVE FAISS INDEX
# ==========================================================

faiss.write_index(
    index,
    str(INDEX_FILE)
)

print(
    f"\nFAISS index saved to:\n{INDEX_FILE}"
)


# ==========================================================
# SAVE METADATA
# ==========================================================

metadata_columns = [
    "chunk_id",
    "document_id",
    "chunk_index",
    "title",
    "chunk_text",
    "word_count",
    "char_count",
    "language",
    "domain",
    "url",
    "source",
]

# Keep only columns that actually exist
metadata_columns = [
    col
    for col in metadata_columns
    if col in df.columns
]

metadata_df = df[
    metadata_columns
].copy()

metadata_df.to_csv(
    METADATA_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nMetadata saved to:\n{METADATA_FILE}"
)


# ==========================================================
# FINAL VERIFICATION
# ==========================================================

saved_index = faiss.read_index(
    str(INDEX_FILE)
)

saved_metadata = pd.read_csv(
    METADATA_FILE,
    encoding="utf-8-sig"
)

assert saved_index.ntotal == len(saved_metadata)

print("\n" + "=" * 60)
print("FAISS INDEX BUILD COMPLETE")
print("=" * 60)

print(
    f"Final passages     : {len(saved_metadata):,}"
)

print(
    f"FAISS vectors      : {saved_index.ntotal:,}"
)

print(
    f"Embedding dimension: {saved_index.d}"
)

print(
    f"Index file         : {INDEX_FILE}"
)

print(
    f"Metadata file      : {METADATA_FILE}"
)

print("=" * 60)
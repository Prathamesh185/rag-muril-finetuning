import faiss
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "models/base_muril"

METADATA_FILE = "data/index/finetuned_metadata.csv"

INDEX_DIR = Path("data/index")
BASE_INDEX_FILE = INDEX_DIR / "base.faiss"

BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256


# ==========================================================
# LOAD CANONICAL CORPUS
# ==========================================================

print("=" * 60)
print("Loading canonical Fine-Tuned FAISS metadata...")
print("=" * 60)

metadata = pd.read_csv(
    METADATA_FILE,
    encoding="utf-8-sig",
)

if "chunk_text" not in metadata.columns:
    raise ValueError(
        "Required column 'chunk_text' not found in metadata."
    )

if metadata["chunk_text"].isna().any():
    raise ValueError(
        "Metadata contains missing chunk_text values."
    )

print(f"Metadata rows : {len(metadata):,}")


# ==========================================================
# VERIFY EXISTING FINE-TUNED INDEX ALIGNMENT
# ==========================================================

FINETUNED_INDEX_FILE = INDEX_DIR / "finetuned.faiss"

finetuned_index = faiss.read_index(
    str(FINETUNED_INDEX_FILE)
)

if finetuned_index.ntotal != len(metadata):
    raise ValueError(
        "Fine-tuned FAISS index and metadata are not aligned."
    )

print(
    f"Fine-tuned vectors : "
    f"{finetuned_index.ntotal:,}"
)


# ==========================================================
# LOAD BASE MURIL
# ==========================================================

print("\n" + "=" * 60)
print("Loading Base MuRIL...")
print("=" * 60)

base_encoder = SentenceTransformer(
    MODEL_NAME
)

base_encoder.max_seq_length = MAX_SEQ_LENGTH

print(
    "Base embedding dimension:",
    base_encoder.get_embedding_dimension(),
)


# ==========================================================
# CREATE BASE EMBEDDINGS
# ==========================================================

print("\n" + "=" * 60)
print("Encoding same corpus using Base MuRIL...")
print("=" * 60)

embeddings = base_encoder.encode(
    metadata["chunk_text"].tolist(),
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

embeddings = embeddings.astype("float32")

print("\nEmbedding shape:", embeddings.shape)


# ==========================================================
# VALIDATION
# ==========================================================

if embeddings.shape[0] != len(metadata):
    raise ValueError(
        "Number of Base embeddings does not match metadata."
    )

dimension = embeddings.shape[1]

if dimension != finetuned_index.d:
    raise ValueError(
        "Base and Fine-Tuned embedding dimensions differ: "
        f"Base={dimension}, "
        f"Fine-Tuned={finetuned_index.d}"
    )


# ==========================================================
# BUILD BASE FAISS INDEX
# ==========================================================

print("\n" + "=" * 60)
print("Building Base MuRIL FAISS index...")
print("=" * 60)

# Normalized embeddings + Inner Product
# = cosine similarity search.
base_index = faiss.IndexFlatIP(
    dimension
)

base_index.add(
    embeddings
)

print(
    f"Base vectors indexed : "
    f"{base_index.ntotal:,}"
)


# ==========================================================
# SAVE
# ==========================================================

faiss.write_index(
    base_index,
    str(BASE_INDEX_FILE),
)

print(
    f"\nBase FAISS index saved to:\n"
    f"{BASE_INDEX_FILE}"
)


# ==========================================================
# FINAL VERIFICATION
# ==========================================================

saved_base_index = faiss.read_index(
    str(BASE_INDEX_FILE)
)

assert saved_base_index.ntotal == len(metadata)
assert saved_base_index.ntotal == finetuned_index.ntotal
assert saved_base_index.d == finetuned_index.d


print("\n" + "=" * 60)
print("BASE MURIL FAISS INDEX BUILD COMPLETE")
print("=" * 60)

print(
    f"Metadata rows       : {len(metadata):,}"
)

print(
    f"Base vectors        : "
    f"{saved_base_index.ntotal:,}"
)

print(
    f"Fine-Tuned vectors  : "
    f"{finetuned_index.ntotal:,}"
)

print(
    f"Embedding dimension : "
    f"{saved_base_index.d}"
)

print(
    f"Base index          : "
    f"{BASE_INDEX_FILE}"
)

print(
    f"Shared metadata     : "
    f"{METADATA_FILE}"
)

print("=" * 60)
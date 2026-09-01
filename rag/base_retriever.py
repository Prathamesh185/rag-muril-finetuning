import faiss
import pandas as pd

from sentence_transformers import SentenceTransformer


# ==========================================================
# CONFIG
# ==========================================================

BASE_MODEL = "models/base_muril"

BASE_INDEX_FILE = "data/index/base.faiss"

METADATA_FILE = "data/index/finetuned_metadata.csv"


# ==========================================================
# LOAD BASE MURIL
# ==========================================================

print("Loading Base MuRIL...")

base_encoder = SentenceTransformer(
    BASE_MODEL
)

base_encoder.max_seq_length = 256

print("Base MuRIL loaded.")
print(
    "Base embedding dimension:",
    base_encoder.get_embedding_dimension()
)


# ==========================================================
# LOAD BASE FAISS + SHARED METADATA
# ==========================================================

print("Loading Base FAISS index...")

base_index = faiss.read_index(
    BASE_INDEX_FILE
)

metadata = pd.read_csv(
    METADATA_FILE,
    encoding="utf-8-sig"
)

print(
    f"Base FAISS vectors : {base_index.ntotal:,}"
)

print(
    f"Metadata rows      : {len(metadata):,}"
)


if base_index.ntotal != len(metadata):
    raise ValueError(
        "Base FAISS index and metadata are not aligned."
    )


# ==========================================================
# RETRIEVE
# ==========================================================

def retrieve_base(question, top_k=5):

    query_embedding = base_encoder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = base_index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1
    ):

        row = metadata.iloc[int(idx)]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "text": row["chunk_text"],
                "source": row["source"],
                "url": row["url"],
            }
        )

    return results
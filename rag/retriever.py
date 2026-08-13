import faiss
import pandas as pd

from rag.config import encoder


# ==========================================================
# CONFIG
# ==========================================================

INDEX_FILE = "data/index/finetuned.faiss"
METADATA_FILE = "data/index/finetuned_metadata.csv"


# ==========================================================
# LOAD FAISS + METADATA
# ==========================================================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)

metadata = pd.read_csv(
    METADATA_FILE,
    encoding="utf-8-sig"
)

print(f"FAISS vectors loaded : {index.ntotal:,}")
print(f"Metadata rows        : {len(metadata):,}")

assert index.ntotal == len(metadata)


# ==========================================================
# RETRIEVE
# ==========================================================

def retrieve(question, top_k=5):

    query_embedding = encoder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
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
                "url": row["url"]
            }
        )

    return results
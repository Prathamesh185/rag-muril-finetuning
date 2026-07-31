import pandas as pd
import numpy as np
import faiss

from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==========================================================
# CONFIG
# ==========================================================

TRAIN_FILE = "data/training/train.csv"
VALID_FILE = "data/training/valid.csv"
TEST_FILE = "data/training/test.csv"

MODEL_PATH = "models/fine_tuned_muril"

OUTPUT_DIR = "data/validation"
OUTPUT_FILE = f"{OUTPUT_DIR}/near_duplicate_passages.csv"

SIMILARITY_THRESHOLD = 0.95

TOP_K = 1

BATCH_SIZE = 16

# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)
print("Loading Fine-Tuned MuRIL...")
print("=" * 60)

model = SentenceTransformer(MODEL_PATH)

# ==========================================================
# LOAD DATA
# ==========================================================

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train : {len(train_df):,}")
print(f"Valid : {len(valid_df):,}")
print(f"Test  : {len(test_df):,}")

# ==========================================================
# REMOVE EXACT DUPLICATES
# ==========================================================

train_df = train_df.drop_duplicates(subset=["positive"]).reset_index(drop=True)
valid_df = valid_df.drop_duplicates(subset=["positive"]).reset_index(drop=True)
test_df = test_df.drop_duplicates(subset=["positive"]).reset_index(drop=True)

print("\nAfter removing exact duplicates")

print(f"Train : {len(train_df):,}")
print(f"Valid : {len(valid_df):,}")
print(f"Test  : {len(test_df):,}")

# ==========================================================
# EMBEDDING FUNCTION
# ==========================================================

def encode(texts):

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")

print("\nEncoding Train Passages...")
train_embeddings = encode(train_df["positive"].tolist())

print("\nBuilding FAISS Index...")

dimension = train_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(train_embeddings)

print(f"Indexed {index.ntotal:,} train passages.")

# ==========================================================
# SEARCH FUNCTION
# ==========================================================

def search_split(split_name, split_df):

    print(f"\nChecking {split_name}...")

    embeddings = encode(split_df["positive"].tolist())

    scores, indices = index.search(
        embeddings,
        TOP_K,
    )

    results = []

    for i in tqdm(range(len(split_df))):

        similarity = float(scores[i][0])

        train_index = int(indices[i][0])

        if similarity >= SIMILARITY_THRESHOLD:

            results.append(
                {
                    "query_split": split_name,
                    "similarity": round(similarity, 4),

                    "query_document":
                        split_df.iloc[i]["document_id"],

                    "train_document":
                        train_df.iloc[train_index]["document_id"],

                    "query_chunk":
                        split_df.iloc[i]["chunk_id"],

                    "train_chunk":
                        train_df.iloc[train_index]["chunk_id"],

                    "query_passage":
                        split_df.iloc[i]["positive"],

                    "train_passage":
                        train_df.iloc[train_index]["positive"],
                }
            )

    return results


# ==========================================================
# SEARCH VALIDATION & TEST
# ==========================================================

all_results = []

all_results.extend(
    search_split(
        "Validation",
        valid_df,
    )
)

all_results.extend(
    search_split(
        "Test",
        test_df,
    )
)

# ==========================================================
# SAVE RESULTS
# ==========================================================

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True,
)

results_df = pd.DataFrame(all_results)

if len(results_df):

    results_df = results_df.sort_values(
        "similarity",
        ascending=False,
    ).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(f"Train passages      : {len(train_df):,}")
print(f"Validation passages : {len(valid_df):,}")
print(f"Test passages       : {len(test_df):,}")

print(f"\nSimilarity Threshold : {SIMILARITY_THRESHOLD}")

print(f"Candidate Duplicates : {len(results_df):,}")

print(f"\nResults saved to:\n{OUTPUT_FILE}")

if len(results_df):

    print("\nTop 10 Most Similar Passages\n")

    print(
        results_df[
            [
                "query_split",
                "similarity",
                "query_chunk",
                "train_chunk",
            ]
        ].head(10)
    )

else:

    print("\nNo near-duplicate passages found.")

print("=" * 60)
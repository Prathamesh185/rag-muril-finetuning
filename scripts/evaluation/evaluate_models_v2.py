import json
import logging
from pathlib import Path

import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim


# ==========================================================
# PATHS AND SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEST_FILE = PROJECT_ROOT / "data" / "training_v2" / "test.csv"

BASE_MODEL = PROJECT_ROOT / "models" / "base_muril"
FINETUNED_MODEL = PROJECT_ROOT / "models" / "fine_tuned_muril_v2"

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "output_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LENGTH = 256
BATCH_SIZE = 64


# ==========================================================
# LOGGING AND DEVICE
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

logger.info(f"Using device: {DEVICE}")


# ==========================================================
# LOAD TEST DATA
# ==========================================================

logger.info("Loading test dataset...")

df = pd.read_csv(
    TEST_FILE,
    encoding="utf-8-sig",
)

df = df.dropna(
    subset=[
        "question",
        "positive",
        "pair_id",
        "chunk_id",
    ]
).reset_index(drop=True)

print(f"\nTest pairs : {len(df):,}")


# ==========================================================
# BUILD INFORMATION RETRIEVAL DATA
# ==========================================================

queries = {}
corpus = {}
relevant_docs = {}

for _, row in df.iterrows():
    query_id = str(row["pair_id"])
    chunk_id = str(row["chunk_id"])

    queries[query_id] = str(row["question"])
    relevant_docs[query_id] = {chunk_id}


unique_passages = df[
    ["chunk_id", "positive"]
].drop_duplicates("chunk_id")

for _, row in unique_passages.iterrows():
    chunk_id = str(row["chunk_id"])
    corpus[chunk_id] = str(row["positive"])


print(f"Queries    : {len(queries):,}")
print(f"Passages   : {len(corpus):,}")


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model(model_path):
    logger.info(f"Loading model: {model_path}")

    model = SentenceTransformer(
        str(model_path),
        device=DEVICE,
    )

    model.max_seq_length = MAX_SEQ_LENGTH

    return model


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(model, model_name):
    logger.info(f"Evaluating: {model_name}")

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,

        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[100],

        batch_size=BATCH_SIZE,

        score_functions={
            "cosine": cos_sim,
        },

        main_score_function="cosine",
        show_progress_bar=True,
        write_csv=False,
        name=model_name,
    )

    return evaluator(model)


# ==========================================================
# EVALUATE BASE MuRIL
# ==========================================================

print("\n" + "=" * 65)
print("EVALUATING BASE MuRIL")
print("=" * 65)

base_model = load_model(BASE_MODEL)

base_results = evaluate_model(
    base_model,
    "base_muril",
)

del base_model

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==========================================================
# EVALUATE FINE-TUNED MuRIL
# ==========================================================

print("\n" + "=" * 65)
print("EVALUATING FINE-TUNED MuRIL")
print("=" * 65)

finetuned_model = load_model(FINETUNED_MODEL)

finetuned_results = evaluate_model(
    finetuned_model,
    "finetuned_muril",
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

base_output = OUTPUT_DIR / "base_results.json"
finetuned_output = OUTPUT_DIR / "finetuned_results.json"

with open(base_output, "w", encoding="utf-8") as file:
    json.dump(
        base_results,
        file,
        indent=4,
        ensure_ascii=False,
    )

with open(finetuned_output, "w", encoding="utf-8") as file:
    json.dump(
        finetuned_results,
        file,
        indent=4,
        ensure_ascii=False,
    )


print("\nEvaluation completed successfully.")

print("\nSaved files:")
print(base_output)
print(finetuned_output)
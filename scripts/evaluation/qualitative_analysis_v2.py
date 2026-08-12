import logging
from pathlib import Path

import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, cos_sim


# ==========================================================
# PATHS AND SETTINGS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TEST_FILE = PROJECT_ROOT / "data" / "training_v2" / "test.csv"

BASE_MODEL = PROJECT_ROOT / "models" / "base_muril"
FINETUNED_MODEL = PROJECT_ROOT / "models" / "fine_tuned_muril_v2"

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "output_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
TOP_K = 10
MAX_SEQ_LENGTH = 256


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
        "chunk_id",
    ]
).reset_index(drop=True)


passage_df = df[
    ["chunk_id", "positive"]
].drop_duplicates("chunk_id").reset_index(drop=True)

questions = df["question"].astype(str).tolist()
passages = passage_df["positive"].astype(str).tolist()
passage_ids = passage_df["chunk_id"].astype(str).tolist()


print(f"\nQuestions : {len(questions):,}")
print(f"Passages  : {len(passages):,}")


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
# ENCODE AND SEARCH
# ==========================================================

def retrieve(model):
    passage_embeddings = model.encode(
        passages,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    query_embeddings = model.encode(
        questions,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return semantic_search(
        query_embeddings,
        passage_embeddings,
        top_k=TOP_K,
        score_function=cos_sim,
    )


# ==========================================================
# BASE MODEL RETRIEVAL
# ==========================================================

print("\n" + "=" * 65)
print("RUNNING BASE MuRIL RETRIEVAL")
print("=" * 65)

base_model = load_model(BASE_MODEL)
base_hits = retrieve(base_model)

del base_model

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==========================================================
# FINE-TUNED MODEL RETRIEVAL
# ==========================================================

print("\n" + "=" * 65)
print("RUNNING FINE-TUNED MuRIL RETRIEVAL")
print("=" * 65)

finetuned_model = load_model(FINETUNED_MODEL)
finetuned_hits = retrieve(finetuned_model)


# ==========================================================
# FIND CORRECT PASSAGE RANK
# ==========================================================

def find_rank(hits, correct_chunk):
    for rank, hit in enumerate(hits, start=1):
        retrieved_chunk = passage_ids[
            hit["corpus_id"]
        ]

        if retrieved_chunk == correct_chunk:
            return rank

    return "Not in Top10"


# ==========================================================
# BUILD QUALITATIVE RESULTS
# ==========================================================

rows = []

for index, row in df.iterrows():
    correct_chunk = str(row["chunk_id"])

    base_rank = find_rank(
        base_hits[index],
        correct_chunk,
    )

    finetuned_rank = find_rank(
        finetuned_hits[index],
        correct_chunk,
    )

    base_top1_hit = base_hits[index][0]
    finetuned_top1_hit = finetuned_hits[index][0]

    base_top1_passage = passages[
        base_top1_hit["corpus_id"]
    ]

    finetuned_top1_passage = passages[
        finetuned_top1_hit["corpus_id"]
    ]

    rows.append(
        {
            "Question": row["question"],
            "Correct Passage": row["positive"],
            "Base Rank": base_rank,
            "Fine-tuned Rank": finetuned_rank,
            "Base Top1 Score": base_top1_hit["score"],
            "Fine-tuned Top1 Score": finetuned_top1_hit["score"],
            "Base Top1 Passage": base_top1_passage,
            "Fine-tuned Top1 Passage": finetuned_top1_passage,
        }
    )


results = pd.DataFrame(rows)


# ==========================================================
# SAVE RESULTS
# ==========================================================

output_file = OUTPUT_DIR / "retrieval_examples.csv"

results.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 65)
print("QUALITATIVE ANALYSIS COMPLETED")
print("=" * 65)

print(f"Examples saved : {len(results):,}")
print(f"Output file    : {output_file}")
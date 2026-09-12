import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)

from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
)

from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)


# ============================================================
# CONFIG
# ============================================================

# IMPORTANT:
# Start Stage-2 training from Fine-Tuned MuRIL V2,
# NOT google/muril-base-cased.
MODEL_NAME = "models/fine_tuned_muril_v2"

# Hard-negative training dataset
TRAIN_FILE = (
    "data/training_v2/"
    "train_hard_negatives_v2_clean.csv"
)

# Keep the same untouched validation split
VALID_FILE = "data/training_v2/valid.csv"

# New V3 output folder
OUTPUT_DIR = "models/muril_mnrl_hardneg_v3"

LOGGING_DIR = f"{OUTPUT_DIR}/logs"

SEED = 42

# Stage-2 should be conservative
NUM_EPOCHS = 1

# Try 32 on Kaggle GPU.
# If CUDA OOM occurs, change only this to 16.
TRAIN_BATCH_SIZE = 32

EVAL_BATCH_SIZE = 64

# Lower than V2 because V2 is already fine-tuned.
LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10

MAX_SEQ_LENGTH = 256

SAVE_STEPS = 200
EVAL_STEPS = 200
LOGGING_STEPS = 100

SAVE_TOTAL_LIMIT = 3


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device: {device}")

if torch.cuda.is_available():
    logger.info(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CHECK PATHS
# ============================================================

if not Path(MODEL_NAME).exists():
    raise FileNotFoundError(
        f"V2 model not found: {MODEL_NAME}"
    )

if not Path(TRAIN_FILE).exists():
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

if not Path(VALID_FILE).exists():
    raise FileNotFoundError(
        f"Validation file not found: {VALID_FILE}"
    )


# ============================================================
# OUTPUT FOLDERS
# ============================================================

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True,
)

Path(LOGGING_DIR).mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

logger.info("=" * 60)
logger.info("Loading V3 hard-negative datasets...")
logger.info("=" * 60)

train_raw = pd.read_csv(
    TRAIN_FILE,
    encoding="utf-8-sig",
)

valid_raw = pd.read_csv(
    VALID_FILE,
    encoding="utf-8-sig",
)

logger.info(
    f"Hard-negative triplets: {len(train_raw):,}"
)

logger.info(
    f"Validation pairs: {len(valid_raw):,}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_train_columns = {
    "question",
    "positive",
    "negative",
}

required_valid_columns = {
    "question",
    "positive",
}

missing_train = (
    required_train_columns - set(train_raw.columns)
)

missing_valid = (
    required_valid_columns - set(valid_raw.columns)
)

if missing_train:
    raise ValueError(
        "Training dataset missing columns: "
        f"{missing_train}"
    )

if missing_valid:
    raise ValueError(
        "Validation dataset missing columns: "
        f"{missing_valid}"
    )


# ============================================================
# PREPARE TRAINING TRIPLETS
# ============================================================

# MNRL receives:
#
# anchor   = question
# positive = correct agriculture passage
# negative = mined hard negative

train_df = train_raw[
    [
        "question",
        "positive",
        "negative",
    ]
].copy()

train_df = train_df.rename(
    columns={
        "question": "anchor",
        "positive": "positive",
        "negative": "negative",
    }
)


# ============================================================
# PREPARE VALIDATION PAIRS
# ============================================================

valid_df = valid_raw[
    [
        "question",
        "positive",
    ]
].copy()

valid_df = valid_df.rename(
    columns={
        "question": "anchor",
        "positive": "positive",
    }
)


# ============================================================
# REMOVE NULL / EMPTY TEXT
# ============================================================

train_df = train_df.dropna(
    subset=[
        "anchor",
        "positive",
        "negative",
    ]
).reset_index(drop=True)

valid_df = valid_df.dropna(
    subset=[
        "anchor",
        "positive",
    ]
).reset_index(drop=True)


# Strip whitespace
for column in [
    "anchor",
    "positive",
    "negative",
]:
    if column in train_df.columns:
        train_df[column] = (
            train_df[column]
            .astype(str)
            .str.strip()
        )

for column in [
    "anchor",
    "positive",
]:
    valid_df[column] = (
        valid_df[column]
        .astype(str)
        .str.strip()
    )


# Remove accidental empty strings
train_df = train_df[
    (train_df["anchor"] != "")
    & (train_df["positive"] != "")
    & (train_df["negative"] != "")
].reset_index(drop=True)

valid_df = valid_df[
    (valid_df["anchor"] != "")
    & (valid_df["positive"] != "")
].reset_index(drop=True)


logger.info(
    f"Training after cleaning: {len(train_df):,}"
)

logger.info(
    f"Validation after cleaning: {len(valid_df):,}"
)


# ============================================================
# SAFETY CHECK:
# POSITIVE SHOULD NOT EQUAL NEGATIVE
# ============================================================

same_text_mask = (
    train_df["positive"]
    .str.casefold()
    .eq(
        train_df["negative"]
        .str.casefold()
    )
)

same_text_count = int(
    same_text_mask.sum()
)

if same_text_count > 0:

    logger.warning(
        f"Removing {same_text_count:,} rows where "
        "positive == negative."
    )

    train_df = train_df[
        ~same_text_mask
    ].reset_index(drop=True)


# ============================================================
# DATASET STATISTICS
# ============================================================

logger.info(
    f"Final training triplets: {len(train_df):,}"
)

logger.info(
    "Unique training questions: "
    f"{train_df['anchor'].nunique():,}"
)

logger.info(
    "Unique positives: "
    f"{train_df['positive'].nunique():,}"
)

logger.info(
    "Unique negatives: "
    f"{train_df['negative'].nunique():,}"
)


# ============================================================
# HUGGINGFACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False,
)

# MNRL also supports ordinary anchor-positive pairs,
# so this can be used for validation loss.
valid_dataset = Dataset.from_pandas(
    valid_df,
    preserve_index=False,
)

logger.info(train_dataset)
logger.info(valid_dataset)


# ============================================================
# LOAD FINE-TUNED MURIL V2
# ============================================================

logger.info("=" * 60)
logger.info("Loading Fine-Tuned MuRIL V2...")
logger.info("=" * 60)

model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)

model.max_seq_length = MAX_SEQ_LENGTH

logger.info(model)


# ============================================================
# LOSS
# ============================================================

# Stage 1:
#   MNRL(question, positive)
#
# Stage 2 / V3:
#   MNRL(question, positive, hard_negative)
#
# The explicit negative is now one of the candidate
# passages the model must learn to rank below the positive.

loss = MultipleNegativesRankingLoss(
    model=model,
)

logger.info(
    "MultipleNegativesRankingLoss initialized "
    "with explicit hard negatives."
)


# ============================================================
# BUILD VALIDATION RETRIEVAL CORPUS
# ============================================================

logger.info("=" * 60)
logger.info("Preparing validation retrieval evaluator...")
logger.info("=" * 60)

queries = {}
corpus = {}
relevant_docs = {}


# ------------------------------------------------------------
# Preferred method:
# use real chunk_id if valid.csv contains it.
# ------------------------------------------------------------

if "chunk_id" in valid_raw.columns:

    logger.info(
        "Using chunk_id for validation corpus."
    )

    valid_eval_df = valid_raw[
        [
            "question",
            "positive",
            "chunk_id",
        ]
    ].dropna().copy()

    valid_eval_df["question"] = (
        valid_eval_df["question"]
        .astype(str)
        .str.strip()
    )

    valid_eval_df["positive"] = (
        valid_eval_df["positive"]
        .astype(str)
        .str.strip()
    )

    valid_eval_df["chunk_id"] = (
        valid_eval_df["chunk_id"]
        .astype(str)
        .str.strip()
    )

    # Build one corpus entry per real passage.
    passage_rows = (
        valid_eval_df[
            [
                "chunk_id",
                "positive",
            ]
        ]
        .drop_duplicates(
            subset=["chunk_id"]
        )
    )

    for _, row in passage_rows.iterrows():

        corpus[
            row["chunk_id"]
        ] = row["positive"]

    # Every question points to its true chunk.
    for idx, row in valid_eval_df.iterrows():

        query_id = f"q{idx}"

        queries[
            query_id
        ] = row["question"]

        relevant_docs[
            query_id
        ] = {
            row["chunk_id"]
        }


# ------------------------------------------------------------
# Fallback:
# if chunk_id is unavailable, deduplicate by passage text.
# ------------------------------------------------------------

else:

    logger.warning(
        "chunk_id not found in valid.csv. "
        "Using unique positive passage text as corpus IDs."
    )

    passage_to_id = {}

    next_doc_id = 0

    for idx, row in valid_df.iterrows():

        passage = row["positive"]

        if passage not in passage_to_id:

            doc_id = f"d{next_doc_id}"

            passage_to_id[
                passage
            ] = doc_id

            corpus[
                doc_id
            ] = passage

            next_doc_id += 1

        query_id = f"q{idx}"

        queries[
            query_id
        ] = row["anchor"]

        relevant_docs[
            query_id
        ] = {
            passage_to_id[passage]
        }


logger.info(
    f"Validation queries: {len(queries):,}"
)

logger.info(
    f"Unique validation passages: {len(corpus):,}"
)


# ============================================================
# INFORMATION RETRIEVAL EVALUATOR
# ============================================================

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="validation",

    # Explicit retrieval cutoffs
    accuracy_at_k=[
        1,
        3,
        5,
        10,
    ],

    precision_recall_at_k=[
        1,
        3,
        5,
        10,
    ],

    mrr_at_k=[
        10,
    ],

    ndcg_at_k=[
        10,
    ],

    map_at_k=[
        100,
    ],

    batch_size=EVAL_BATCH_SIZE,

    show_progress_bar=True,
)

logger.info(
    "InformationRetrievalEvaluator created."
)


# ============================================================
# BASELINE V2 VALIDATION
# ============================================================

# Useful because the same evaluator is used before and after
# Stage-2 training.

logger.info("=" * 60)
logger.info("Evaluating V2 before Stage-2 training...")
logger.info("=" * 60)

v2_results = ir_evaluator(model)

print("\n")
print("=" * 70)
print("V2 VALIDATION RESULTS BEFORE HARD-NEGATIVE TRAINING")
print("=" * 70)

for metric, value in v2_results.items():

    if isinstance(value, float):

        print(
            f"{metric:<55} "
            f"{value:.6f}"
        )

print("=" * 70)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = SentenceTransformerTrainingArguments(

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Stage-2 epochs
    # --------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_ratio=WARMUP_RATIO,

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    fp16=torch.cuda.is_available(),

    bf16=False,

    # --------------------------------------------------------
    # MNRL
    # --------------------------------------------------------

    # Avoid duplicated anchors/positives/negatives in
    # the same batch whenever possible.
    batch_sampler=BatchSamplers.NO_DUPLICATES,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=EVAL_STEPS,

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=SAVE_STEPS,

    save_total_limit=SAVE_TOTAL_LIMIT,

    # At the end, restore checkpoint with best MRR@10.
    load_best_model_at_end=True,

    metric_for_best_model=(
        "eval_validation_cosine_mrr@10"
    ),

    greater_is_better=True,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=LOGGING_STEPS,

    logging_first_step=True,

    logging_dir=LOGGING_DIR,

    report_to="tensorboard",

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader_num_workers=4,

    dataloader_pin_memory=True,

    remove_unused_columns=False,

    # --------------------------------------------------------
    # Run name
    # --------------------------------------------------------

    run_name="muril_mnrl_hardneg_v3",
)

logger.info(
    "Training arguments created."
)


# ============================================================
# TRAINER
# ============================================================

trainer = SentenceTransformerTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    # Pair validation dataset is valid for MNRL.
    eval_dataset=valid_dataset,

    loss=loss,

    evaluator=ir_evaluator,
)

logger.info(
    "SentenceTransformerTrainer initialized."
)


# ============================================================
# CHECKPOINT RESUME
# ============================================================

resume_checkpoint = None

checkpoints = sorted(

    Path(OUTPUT_DIR).glob(
        "checkpoint-*"
    ),

    key=lambda x: int(
        x.name.split("-")[-1]
    ),
)

if checkpoints:

    resume_checkpoint = str(
        checkpoints[-1]
    )

    logger.info(
        f"Resuming from checkpoint: "
        f"{resume_checkpoint}"
    )

else:

    logger.info(
        "No V3 checkpoint found. "
        "Starting Stage-2 training from V2."
    )


# ============================================================
# TRAIN
# ============================================================

logger.info("=" * 60)
logger.info(
    "STARTING MURIL V3 HARD-NEGATIVE TRAINING"
)
logger.info("=" * 60)

trainer.train(
    resume_from_checkpoint=resume_checkpoint
)

logger.info("=" * 60)
logger.info("TRAINING FINISHED")
logger.info("=" * 60)


# ============================================================
# SAVE FINAL / BEST MODEL
# ============================================================

FINAL_MODEL_DIR = (
    f"{OUTPUT_DIR}/final"
)

Path(
    FINAL_MODEL_DIR
).mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(
    FINAL_MODEL_DIR
)

# Save complete SentenceTransformer format
model.save(
    FINAL_MODEL_DIR
)

logger.info(
    f"V3 model saved to: "
    f"{FINAL_MODEL_DIR}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

logger.info("=" * 60)
logger.info(
    "Running final V3 validation evaluation..."
)
logger.info("=" * 60)

v3_results = ir_evaluator(
    model
)


print("\n")
print("=" * 70)
print("FINAL V3 VALIDATION RESULTS")
print("=" * 70)

for metric, value in v3_results.items():

    if isinstance(value, float):

        print(
            f"{metric:<55} "
            f"{value:.6f}"
        )

print("=" * 70)


# ============================================================
# V2 VS V3 VALIDATION COMPARISON
# ============================================================

print("\n")
print("=" * 85)
print("V2 VS V3 VALIDATION COMPARISON")
print("=" * 85)

print(
    f"{'Metric':<55}"
    f"{'V2':>14}"
    f"{'V3':>14}"
)

print("-" * 85)

all_metrics = sorted(
    set(v2_results.keys())
    | set(v3_results.keys())
)

for metric in all_metrics:

    v2_value = v2_results.get(metric)
    v3_value = v3_results.get(metric)

    if (
        isinstance(v2_value, float)
        and isinstance(v3_value, float)
    ):

        print(
            f"{metric:<55}"
            f"{v2_value:>14.6f}"
            f"{v3_value:>14.6f}"
        )

print("=" * 85)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = (
    Path(OUTPUT_DIR)
    / "validation_metrics_v2_vs_v3.txt"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "=" * 80
        + "\n"
    )

    f.write(
        "V2 VALIDATION RESULTS "
        "BEFORE HARD-NEGATIVE TRAINING\n"
    )

    f.write(
        "=" * 80
        + "\n"
    )

    for metric, value in v2_results.items():

        f.write(
            f"{metric}: "
            f"{value}\n"
        )

    f.write("\n")

    f.write(
        "=" * 80
        + "\n"
    )

    f.write(
        "V3 VALIDATION RESULTS "
        "AFTER HARD-NEGATIVE TRAINING\n"
    )

    f.write(
        "=" * 80
        + "\n"
    )

    for metric, value in v3_results.items():

        f.write(
            f"{metric}: "
            f"{value}\n"
        )


logger.info(
    f"Validation metrics saved to: "
    f"{metrics_path}"
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MURIL V3 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Starting model       : "
    f"{MODEL_NAME}"
)

print(
    f"Training triplets    : "
    f"{len(train_dataset):,}"
)

print(
    f"Validation queries   : "
    f"{len(queries):,}"
)

print(
    f"Validation passages  : "
    f"{len(corpus):,}"
)

print(
    f"Epochs               : "
    f"{NUM_EPOCHS}"
)

print(
    f"Batch size           : "
    f"{TRAIN_BATCH_SIZE}"
)

print(
    f"Learning rate        : "
    f"{LEARNING_RATE}"
)

print(
    f"Max sequence length  : "
    f"{MAX_SEQ_LENGTH}"
)

print(
    "\nLoss:"
)

print(
    "MultipleNegativesRankingLoss "
    "+ explicit hard negatives"
)

print(
    f"\nFinal model saved to:\n"
    f"{FINAL_MODEL_DIR}"
)

print("=" * 70)
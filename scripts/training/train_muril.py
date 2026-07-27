import os
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch    # deep learning framework

from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
)

from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)

from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
)

# ==========================================================
# CONFIG
# ==========================================================

MODEL_NAME = "google/muril-base-cased"

TRAIN_FILE = "data/training/train.csv"
VALID_FILE = "data/training/valid.csv"

# Folder to save trained muril
OUTPUT_DIR = "models/muril_mnrl"

CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"

LOGGING_DIR = "logs"

SEED = 42

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64

LEARNING_RATE = 2e-5  # Rate at which model should learn (how much the model updates its weights after each step)

WEIGHT_DECAY = 0.01  # helps prevent overfitting

WARMUP_RATIO = 0.10  # Training starts gradually

MAX_SEQ_LENGTH = 256 # How much text can MuRIL read at once

SAVE_STEPS = 200     # save progress regularly

EVAL_STEPS = 200     # Evaluate model performance every 500 training steps

LOGGING_STEPS = 100  # Log training progress every 100 steps

SAVE_TOTAL_LIMIT = 3 # Keep only the latest 3 checkpoints to save disk space

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

# ==========================================================
# RANDOM SEED
# ==========================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
# DEVICE
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Device : {device}")

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGGING_DIR).mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

logger.info("Loading datasets...")

train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")

valid_df = pd.read_csv(VALID_FILE, encoding="utf-8-sig")

logger.info(f"Training pairs  : {len(train_df):,}")
logger.info(f"Validation pairs: {len(valid_df):,}")

# ==========================================================
# KEEP ONLY REQUIRED COLUMNS
# ==========================================================

train_df = train_df[
    [
        "question",
        "positive",
    ]
].copy()

valid_df = valid_df[
    [
        "question",
        "positive",
    ]
].copy()

train_df = train_df.rename(
    columns={
        "question": "anchor",
        "positive": "positive",
    }
)

valid_df = valid_df.rename(
    columns={
        "question": "anchor",
        "positive": "positive",
    }
)

# ==========================================================
# REMOVE NULLS
# ==========================================================

train_df = train_df.dropna().reset_index(drop=True)

valid_df = valid_df.dropna().reset_index(drop=True)

logger.info(f"Training after cleaning : {len(train_df):,}")
logger.info(f"Validation after cleaning : {len(valid_df):,}")

# ==========================================================
# HUGGINGFACE DATASETS
# ==========================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False,
)

valid_dataset = Dataset.from_pandas(
    valid_df,
    preserve_index=False,
)

logger.info(train_dataset)
logger.info(valid_dataset)

# ==========================================================
# LOAD MURIL
# ==========================================================

logger.info("Loading MuRIL...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)

model.max_seq_length = MAX_SEQ_LENGTH   # limits the maximum number of input tokens.(control GPU memory usage and training speed.) 

logger.info(model)

# ==========================================================
# LOSS
# ==========================================================

loss = MultipleNegativesRankingLoss(model)

logger.info("MultipleNegativesRankingLoss initialized.")

print("=" * 60)
print("Everything Loaded Successfully")
print("=" * 60)
print(f"Model           : {MODEL_NAME}")
print(f"Train pairs     : {len(train_dataset):,}")
print(f"Validation pairs: {len(valid_dataset):,}")
print("=" * 60)

#-------------------------------------------------------
# PART 2
#-------------------------------------------------------

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
)

# ==========================================================
# BUILD VALIDATION CORPUS
# ==========================================================

logger.info("Preparing validation evaluator...")

queries = {}
corpus = {}
relevant_docs = {}

for idx, row in valid_df.iterrows():

    query_id = f"q{idx}"
    doc_id = f"d{idx}"

    queries[query_id] = row["anchor"]

    corpus[doc_id] = row["positive"]

    relevant_docs[query_id] = {doc_id}

logger.info(f"Queries : {len(queries):,}")
logger.info(f"Corpus  : {len(corpus):,}")

# ==========================================================
# INFORMATION RETRIEVAL EVALUATOR
# ==========================================================

ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="validation",
)

logger.info("Evaluator created.")

# ==========================================================
# TRAINING ARGUMENTS
# ==========================================================

training_args = SentenceTransformerTrainingArguments(

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    output_dir=OUTPUT_DIR,
    
    # ------------------------------------------------------
    # Epochs
    # ------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    # ------------------------------------------------------
    # Batch size
    # ------------------------------------------------------

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_steps=0.10,

    # ------------------------------------------------------
    # Mixed precision
    # ------------------------------------------------------

    fp16=torch.cuda.is_available(),

    bf16=False,

    # ------------------------------------------------------
    # MNRL requirement
    # ------------------------------------------------------

    batch_sampler=BatchSamplers.NO_DUPLICATES,

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    eval_strategy="steps",

    eval_steps=EVAL_STEPS,

    # ------------------------------------------------------
    # Saving
    # ------------------------------------------------------

    save_strategy="steps",

    save_steps=SAVE_STEPS,

    save_total_limit=SAVE_TOTAL_LIMIT,

    load_best_model_at_end=True,

    # InformationRetrievalEvaluator reports retrieval metrics;
    # we'll use MRR@10 to choose the best checkpoint.
    metric_for_best_model="eval_validation_cosine_mrr@10",

    greater_is_better=True,

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------

    logging_strategy="steps",

    logging_steps=LOGGING_STEPS,

    logging_first_step=True,

    logging_dir=LOGGING_DIR,

    report_to="tensorboard",

    # ------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------

    seed=SEED,

    # ------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------

    dataloader_num_workers=4,

    dataloader_pin_memory=True,

    remove_unused_columns=False,

    run_name="muril_mnrl",
)

logger.info("Training arguments created.")

# ==========================================================
# TRAINER
# ==========================================================

trainer = SentenceTransformerTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=valid_dataset,

    loss=loss,

    evaluator=ir_evaluator,

)

logger.info("Trainer initialized.")

# ==========================================================
# CHECKPOINT RESUME
# ==========================================================

resume_checkpoint = None

checkpoints = sorted(
    Path(OUTPUT_DIR).glob("checkpoint-*"),
    key=lambda x: int(x.name.split("-")[-1]),
)

if len(checkpoints) > 0:

    resume_checkpoint = str(checkpoints[-1])

    logger.info(f"Resuming from: {resume_checkpoint}")

else:

    logger.info("No checkpoint found. Starting fresh.")


#-------------------------------------------------------------
# PART 3
#-------------------------------------------------------------

# ==========================================================
# TRAIN
# ==========================================================

logger.info("=" * 60)
logger.info("Starting Training...")
logger.info("=" * 60)

trainer.train(
    resume_from_checkpoint=resume_checkpoint
)

logger.info("=" * 60)
logger.info("Training Finished")
logger.info("=" * 60)

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

FINAL_MODEL_DIR = f"{OUTPUT_DIR}/final"

Path(FINAL_MODEL_DIR).mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(FINAL_MODEL_DIR)

logger.info(f"Final model saved to:")
logger.info(FINAL_MODEL_DIR)

# ==========================================================
# SAVE TOKENIZER + CONFIG
# ==========================================================

model.save(FINAL_MODEL_DIR)

logger.info("SentenceTransformer model saved.")

# ==========================================================
# FINAL EVALUATION
# ==========================================================

logger.info("=" * 60)
logger.info("Running Final Validation Evaluation...")
logger.info("=" * 60)

results = ir_evaluator(model)

print("\n")
print("=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)

for metric, value in results.items():

    if isinstance(value, float):

        print(f"{metric:<45} {value:.6f}")

    else:

        print(f"{metric:<45} {value}")

print("=" * 70)

# ==========================================================
# SAVE METRICS
# ==========================================================

metrics_path = Path(OUTPUT_DIR) / "validation_metrics.txt"

with open(metrics_path, "w", encoding="utf-8") as f:

    f.write("=" * 70 + "\n")
    f.write("FINAL VALIDATION RESULTS\n")
    f.write("=" * 70 + "\n")

    for metric, value in results.items():

        f.write(f"{metric}: {value}\n")

logger.info(f"Metrics saved to {metrics_path}")

# ==========================================================
# SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"Training pairs      : {len(train_dataset):,}")
print(f"Validation pairs    : {len(valid_dataset):,}")

print(f"Epochs              : {NUM_EPOCHS}")
print(f"Batch size          : {TRAIN_BATCH_SIZE}")
print(f"Learning rate       : {LEARNING_RATE}")

print(f"\nModel saved to:")
print(FINAL_MODEL_DIR)

print("\nValidation metrics:")
for metric, value in results.items():

    if isinstance(value, float):
        print(f"{metric:<45} {value:.6f}")

print("=" * 70)
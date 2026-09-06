from typing import Literal
from functools import lru_cache
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(
        "Preloading AgriSahayak models and FAISS indexes..."
    )

    from rag.retriever import retrieve
    from rag.base_retriever import retrieve_base

    warmup_question = (
        "गेहूं की सिंचाई कब करनी चाहिए?"
    )

    try:
        retrieve(
            warmup_question,
            top_k=1,
        )

        retrieve_base(
            warmup_question,
            top_k=1,
        )

        print(
            "Models and FAISS indexes preloaded successfully."
        )

    except Exception as exc:
        print(
            f"Startup preload failed: {exc}"
        )

    yield

    print(
        "AgriSahayak API shutting down."
    )


app = FastAPI(
    title="AgriSahayak API",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# REQUEST / RESPONSE MODELS
# ==========================================================

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    model_choice: Literal[
        "Local Qwen",
        "Gemini API",
    ]


class RetrievedPassage(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    title: str
    text: str
    source: str
    url: str


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class CompareRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class LabeledAnswerRequest(BaseModel):
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedPassage]


class RetrieveResponse(BaseModel):
    retrieved: list[RetrievedPassage]


class CompareResponse(BaseModel):
    base: list[RetrievedPassage]
    finetuned: list[RetrievedPassage]

    ground_truth_available: bool
    ground_truth_in_live_corpus: bool
    ground_truth_chunk_id: str | None = None

    base_rank: int | None = None
    finetuned_rank: int | None = None

    rank_cutoff: int = 20


class LabeledAnswerResponse(BaseModel):
    available: bool
    answer: str | None = None
    supporting_text: str | None = None
    positive_passage: str | None = None


# ==========================================================
# BASIC ENDPOINTS
# ==========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    from rag.pipeline import rag_answer

    return rag_answer(
        request.question,
        request.model_choice,
    )


@app.post(
    "/api/retrieve",
    response_model=RetrieveResponse,
)
def retrieve_passages(
    request: RetrieveRequest,
):
    from rag.retriever import retrieve

    results = retrieve(
        request.question,
        top_k=request.top_k,
    )

    return {
        "retrieved": results
    }


# ==========================================================
# DATA FILES
# ==========================================================

TEST_FILE = Path(
    "data/training_v2/test.csv"
)

LIVE_METADATA_FILE = Path(
    "data/index/finetuned_metadata.csv"
)


# ==========================================================
# GROUND-TRUTH LOOKUPS
# ==========================================================

@lru_cache(maxsize=1)
def load_ground_truth_lookup():
    df = pd.read_csv(
        TEST_FILE,
        usecols=[
            "question",
            "chunk_id",
        ],
    )

    lookup = {}

    for _, row in df.iterrows():
        question = str(
            row["question"]
        ).strip()

        chunk_id = str(
            row["chunk_id"]
        ).strip()

        if question and chunk_id:
            lookup[question] = chunk_id

    return lookup


@lru_cache(maxsize=1)
def load_labeled_passage_lookup():
    df = pd.read_csv(
        TEST_FILE,
        usecols=[
            "question",
            "positive",
        ],
    )

    lookup = {}

    for _, row in df.iterrows():
        question = str(
            row["question"]
        ).strip()

        positive = str(
            row["positive"]
        ).strip()

        if question and positive:
            lookup[question] = positive

    return lookup


@lru_cache(maxsize=1)
def load_live_chunk_ids():
    df = pd.read_csv(
        LIVE_METADATA_FILE,
        usecols=[
            "chunk_id",
        ],
    )

    return set(
        df["chunk_id"]
        .astype(str)
        .str.strip()
    )


def get_ground_truth_chunk_id(
    question: str,
):
    lookup = load_ground_truth_lookup()

    return lookup.get(
        question.strip()
    )


def get_labeled_positive_passage(
    question: str,
):
    lookup = load_labeled_passage_lookup()

    return lookup.get(
        question.strip()
    )


def find_result_rank(
    results,
    chunk_id,
):
    if not chunk_id:
        return None

    for item in results:
        if item["chunk_id"] == chunk_id:
            return item["rank"]

    return None


# ==========================================================
# MODEL COMPARISON
# ==========================================================

@app.post(
    "/api/compare",
    response_model=CompareResponse,
)
def compare_models(
    request: CompareRequest,
):
    from rag.base_retriever import (
        retrieve_base,
    )
    from rag.retriever import retrieve

    ground_truth_chunk_id = (
        get_ground_truth_chunk_id(
            request.question
        )
    )

    ground_truth_available = (
        ground_truth_chunk_id is not None
    )

    ground_truth_in_live_corpus = (
        ground_truth_available
        and ground_truth_chunk_id
        in load_live_chunk_ids()
    )

    rank_cutoff = 20

    search_top_k = (
        rank_cutoff
        if ground_truth_in_live_corpus
        else request.top_k
    )

    base_all = retrieve_base(
        request.question,
        top_k=search_top_k,
    )

    finetuned_all = retrieve(
        request.question,
        top_k=search_top_k,
    )

    base_rank = (
        find_result_rank(
            base_all,
            ground_truth_chunk_id,
        )
        if ground_truth_in_live_corpus
        else None
    )

    finetuned_rank = (
        find_result_rank(
            finetuned_all,
            ground_truth_chunk_id,
        )
        if ground_truth_in_live_corpus
        else None
    )

    return {
        # UI receives only requested Top-K passages
        "base":
            base_all[:request.top_k],

        "finetuned":
            finetuned_all[:request.top_k],

        "ground_truth_available":
            ground_truth_available,

        "ground_truth_in_live_corpus":
            ground_truth_in_live_corpus,

        "ground_truth_chunk_id":
            ground_truth_chunk_id,

        "base_rank":
            base_rank,

        "finetuned_rank":
            finetuned_rank,

        "rank_cutoff":
            rank_cutoff,
    }


# ==========================================================
# GENERATED ANSWER FROM LABELED PASSAGE
# ==========================================================

@app.post(
    "/api/labeled-answer",
    response_model=LabeledAnswerResponse,
)
def generate_labeled_answer(
    request: LabeledAnswerRequest,
):
    positive_passage = (
        get_labeled_positive_passage(
            request.question
        )
    )

    if not positive_passage:
        return {
            "available": False,
            "answer": None,
            "supporting_text": None,
            "positive_passage": None,
        }

    from rag.llm import gemini_llm

    prompt = f"""
You are given one agriculture question and its labeled positive passage.

Your job is to produce TWO outputs.

1. ANSWER:
Give a short and direct answer to the question using ONLY the labeled passage.

2. SUPPORT:
Copy the exact sentence or shortest exact text span from the labeled passage
that directly supports the answer.

Rules:
- Do not use outside knowledge.
- Do not add information that is not present in the passage.
- Keep the answer in the same language as the question.
- SUPPORT must be copied exactly from the passage.
- Do not paraphrase SUPPORT.
- Keep SUPPORT as short as possible while still supporting the answer.
- If the passage does not clearly contain a direct answer, say so in ANSWER
  and leave SUPPORT empty.

Return exactly in this format:

ANSWER: <short answer>
SUPPORT: <exact text copied from passage>

Question:
{request.question}

Labeled passage:
{positive_passage}
"""

    raw_output = gemini_llm(
        prompt
    ).strip()

    answer = raw_output
    supporting_text = ""

    # Parse Gemini response
    if "SUPPORT:" in raw_output:
        answer_part, support_part = (
            raw_output.split(
                "SUPPORT:",
                1,
            )
        )

        answer = (
            answer_part
            .replace(
                "ANSWER:",
                "",
                1,
            )
            .strip()
        )

        supporting_text = (
            support_part.strip()
        )

    else:
        answer = (
            raw_output
            .replace(
                "ANSWER:",
                "",
                1,
            )
            .strip()
        )

    # Safety check:
    # Only highlight text that truly exists
    # inside the labeled positive passage.
    if (
        supporting_text
        and supporting_text
        not in positive_passage
    ):
        supporting_text = ""

    return {
        "available": True,
        "answer": answer,
        "supporting_text":
            supporting_text,
        "positive_passage":
            positive_passage,
    }
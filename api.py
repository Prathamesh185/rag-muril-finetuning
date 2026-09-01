from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="AgriSahayak API",
    version="1.0.0",
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


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    model_choice: Literal["Local Qwen", "Gemini API"]


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
    top_k: int = Field(default=5, ge=1, le=20)

class CompareRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

class ChatResponse(BaseModel):
    answer: str
    retrieved: list[RetrievedPassage]

class RetrieveResponse(BaseModel):
    retrieved: list[RetrievedPassage]

class CompareResponse(BaseModel):
    base: list[RetrievedPassage]
    finetuned: list[RetrievedPassage]

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    from rag.pipeline import rag_answer

    return rag_answer(
        request.question,
        request.model_choice,
    )

@app.post("/api/retrieve", response_model=RetrieveResponse)
def retrieve_passages(request: RetrieveRequest):
    from rag.retriever import retrieve

    results = retrieve(
        request.question,
        top_k=request.top_k,
    )

    return {
        "retrieved": results
    }

@app.post("/api/compare", response_model=CompareResponse)
def compare_models(request: CompareRequest):
    from rag.base_retriever import retrieve_base
    from rag.retriever import retrieve

    base_results = retrieve_base(
        request.question,
        top_k=request.top_k,
    )

    finetuned_results = retrieve(
        request.question,
        top_k=request.top_k,
    )

    return {
        "base": base_results,
        "finetuned": finetuned_results,
    }
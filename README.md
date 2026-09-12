# 🌾 Agriculture-Aware RAG using Fine-Tuned MuRIL

This project fine-tunes Google's **MuRIL model as an agriculture-aware sentence encoder** to improve domain-specific semantic retrieval for Indian-language Retrieval-Augmented Generation (RAG).

The current implementation focuses primarily on **Hindi agriculture data**. The research pipeline uses **Fine-Tuned MuRIL V3, FAISS, Gemini API / Local Qwen, FastAPI, and React** to retrieve relevant agriculture passages and generate grounded answers.

> **Core Focus:** Fine-tuning MuRIL for agriculture-specific semantic retrieval and evaluating its improvement over Base MuRIL and strong multilingual embedding baselines.

---

## ✨ Highlights

- Agriculture-specific semantic retrieval using Fine-Tuned MuRIL
- Hindi agriculture question–passage dataset from Vikaspedia
- Stage-1 MNRL fine-tuning + Stage-2 hard-negative refinement
- FAISS-based dense retrieval with grounded RAG
- Gemini API / Local Qwen integration
- React + FastAPI web application
- Comparison with Base MuRIL, E5-base, and BGE-M3
- Duplicate and leakage validation

---

## ✅ Project Status

### Completed

- [x] Built and validated a Hindi agriculture dataset with **20,141 question–passage pairs**
- [x] Fine-tuned MuRIL using MNRL to create **MuRIL V2**
- [x] Performed hard-negative mining and Stage-2 refinement to create **MuRIL V3**
- [x] Evaluated Base MuRIL, V2, V3, E5-base, and BGE-M3
- [x] Built FAISS retrieval, RAG pipeline, FastAPI backend, and React frontend
- [x] Added AI Assistant, Retrieval Analysis, and Model Comparison interfaces

### Future Work

- [ ] Extend to Kannada and other Indian languages
- [ ] Build an independent external agriculture test set
- [ ] Explore additional contrastive objectives
- [ ] Public deployment / Hugging Face model release

---

## 🏗️ Project Pipeline

### Training and Dataset Pipeline

```text
Vikaspedia Hindi Agriculture Articles
        ↓
Data Collection
        ↓
Cleaning & Preprocessing
        ↓
Sentence-Based Chunking
        ↓
Question Generation
        ↓
Question–Passage Dataset
        ↓
Duplicate & Leakage Validation
        ↓
Train / Validation / Test Split
        ↓
Base MuRIL
        ↓
Stage-1 Fine-Tuning using MNRL
        ↓
Fine-Tuned MuRIL V2
        ↓
Hard-Negative Mining
        ↓
Hard-Negative Cleaning / Quality Review
        ↓
Stage-2 Fine-Tuning using MNRL + Hard Negatives
        ↓
Fine-Tuned MuRIL V3
```

---

### Runtime RAG Pipeline

> Fine-Tuned MuRIL V3 evaluation is complete. Integration of V3 into the live application FAISS index is currently in progress.

```text
User Question
      ↓
Fine-Tuned MuRIL Encoder
      ↓
768-D Query Embedding
      ↓
FAISS Index
      ↓
Top-K Agriculture Passages
      ↓
Grounded Context
      ↓
Gemini API / Local Qwen
      ↓
Hindi Answer + Retrieved Sources
```

---

### Base vs Fine-Tuned Retrieval Comparison

```text
                    ┌── Base MuRIL ───────→ Base FAISS ───────┐
User Question ──────┤                                         ├──→ Compare Top-K Results
                    └── Fine-Tuned MuRIL ─→ Fine-Tuned FAISS ─┘
```

Both retrieval paths use the same passage corpus and metadata ordering so that the encoder remains the main variable being compared.

---

## 📊 Evaluation

The project evaluates the following retrieval models:

- **Base MuRIL**
- **Fine-Tuned MuRIL V2**
- **Fine-Tuned MuRIL V3**
- **E5-base**
- **BGE-M3**

All models are evaluated on the same held-out Hindi agriculture retrieval test set.

### Test Set

- **1,980 queries**
- **744 unique passages**

---

## 📈 Final Retrieval Results

| Metric | Base MuRIL | Fine-Tuned MuRIL V2 | Fine-Tuned MuRIL V3 | E5-base | BGE-M3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accuracy@1 | 0.2146 | 0.7010 | **0.7490** | 0.7328 | 0.6975 |
| Accuracy@3 | 0.3389 | 0.8803 | **0.9071** | 0.8949 | 0.8631 |
| Accuracy@5 | 0.3939 | 0.9318 | **0.9434** | 0.9303 | 0.9086 |
| Accuracy@10 | 0.4884 | 0.9662 | **0.9732** | 0.9611 | 0.9490 |
| Precision@1 | 0.2146 | 0.7010 | **0.7490** | 0.7328 | 0.6975 |
| Precision@3 | 0.1130 | 0.2934 | **0.3024** | 0.2983 | 0.2877 |
| Precision@5 | 0.0788 | 0.1864 | **0.1887** | 0.1861 | 0.1817 |
| Precision@10 | 0.0488 | 0.0966 | **0.0973** | 0.0961 | 0.0949 |
| Recall@1 | 0.2146 | 0.7010 | **0.7490** | 0.7328 | 0.6975 |
| Recall@3 | 0.3389 | 0.8803 | **0.9071** | 0.8949 | 0.8631 |
| Recall@5 | 0.3939 | 0.9318 | **0.9434** | 0.9303 | 0.9086 |
| Recall@10 | 0.4884 | 0.9662 | **0.9732** | 0.9611 | 0.9490 |
| MRR@10 | 0.2919 | 0.7999 | **0.8331** | 0.8196 | 0.7889 |
| nDCG@10 | 0.3383 | 0.8410 | **0.8677** | 0.8546 | 0.8281 |
| MAP@100 | 0.3063 | 0.8013 | **0.8343** | 0.8213 | 0.7911 |

Fine-Tuned MuRIL V3 achieved the strongest result across all reported metrics on the current held-out test set.

---

## 📊 V2 → V3 Improvement

Hard-negative refinement further improved retrieval performance.

| Metric | V2 | V3 | Improvement |
| --- | ---: | ---: | ---: |
| Accuracy@1 | 0.7010 | **0.7490** | **+0.0480** |
| Accuracy@3 | 0.8803 | **0.9071** | **+0.0268** |
| Accuracy@5 | 0.9318 | **0.9434** | **+0.0116** |
| Accuracy@10 | 0.9662 | **0.9732** | **+0.0070** |
| MRR@10 | 0.7999 | **0.8331** | **+0.0332** |
| nDCG@10 | 0.8410 | **0.8677** | **+0.0268** |
| MAP@100 | 0.8013 | **0.8343** | **+0.0330** |

### Key Result

```text
Accuracy@1
70.10% → 74.90%
```

The Stage-2 hard-negative refinement improved both top-rank accuracy and overall retrieval ranking quality.

---

## 🧠 Fine-Tuning Strategy

### Stage 1 — MuRIL V2

The first stage fine-tunes MuRIL using agriculture question–positive passage pairs.

| Setting | Value |
| --- | --- |
| Base model | `google/muril-base-cased` |
| Objective | Multiple Negatives Ranking Loss |
| Epochs | 3 |
| Learning rate | `2e-5` |
| Max sequence length | 256 |
| Embedding dimension | 768 |
| Training pairs | 16,292 |
| Validation pairs | 1,869 |

Stage-1 output:

```text
Fine-Tuned MuRIL V2
```

---

### Stage 2 — Hard-Negative Refinement

Fine-Tuned MuRIL V2 is used to retrieve difficult but incorrect passages from the training corpus.

Hard-negative candidates are filtered to remove:

- Positive passages
- Other known positives
- Same-document candidates
- Same duplicate/split-group candidates
- Exact duplicate passages
- Near-duplicate passages
- Candidates that are too close to the positive
- Candidates that are too easy

Initial mining produced:

```text
30,597 hard-negative triplets
```

After review and cleaning:

```text
30,585 training triplets
```

Stage-2 training starts from the already fine-tuned MuRIL V2 model.

| Setting | Value |
| --- | --- |
| Starting model | Fine-Tuned MuRIL V2 |
| Objective | MNRL with explicit hard negatives |
| Epochs | 1 |
| Batch size | 32 |
| Learning rate | `1e-5` |
| Max sequence length | 256 |
| Training triplets | 30,585 |
| Validation pairs | 1,869 |
| Selected checkpoint | Step 478 |

Stage-2 output:

```text
Fine-Tuned MuRIL V3
```

---

## 🔍 Why Hard Negatives?

Random or in-batch negatives can become too easy for an already fine-tuned retriever.

Hard negatives are passages that are:

- semantically similar to the query,
- retrieved highly by the current model,
- but still incorrect.

Training with these difficult examples encourages the encoder to learn finer semantic distinctions.

This is particularly useful in agriculture because many passages contain overlapping terminology related to crops, fertilizers, irrigation, diseases, soil conditions, cultivation practices, and treatments.

---

## 🧹 Dataset Validation

Before final training and evaluation, the dataset was checked for:

- Exact duplicate documents
- Near-duplicate documents
- Exact duplicate questions
- Near-duplicate questions
- Question-to-passage copying
- Train / validation / test leakage
- Duplicate-document grouping across splits

### Final Dataset

- **20,141 question–passage pairs**
- **7,379 unique chunks**
- **1,814 documents**
- **1,770 split groups**

### Split

```text
Train      : 16,292
Validation : 1,869
Test       : 1,980
```

The held-out test split remained unchanged during Stage-2 hard-negative training.

---

## 🧠 Why MuRIL?

MuRIL was selected because it was specifically developed for Indian languages and supports multilingual and transliterated Indian-language text.

The objective of this project is not simply to use MuRIL directly, but to adapt it to the **agriculture domain** so that relevant agriculture passages are ranked more accurately for user queries.

The project demonstrates a progressive retrieval improvement:

```text
Base MuRIL
      ↓
Domain Fine-Tuning
      ↓
MuRIL V2
      ↓
Hard-Negative Refinement
      ↓
MuRIL V3
```

---

## 🌐 Web Application

The application contains three main interfaces.

### 1. AI Assistant

Users can ask agriculture questions in Hindi and receive:

- Grounded answers
- Retrieved evidence passages
- Cosine similarity scores
- Source names
- Source URLs

Users can choose between:

- **Gemini API**
- **Local Qwen through Ollama**

---

### 2. Retrieval Analysis

Shows the real retrieval pipeline:

```text
User Query
→ Fine-Tuned MuRIL
→ Query Embedding
→ FAISS
→ Top-K Passages
```

It displays the retrieved agriculture passages before LLM generation.

---

### 3. Model Comparison

Runs the same question through:

- Base MuRIL
- Fine-Tuned MuRIL

and displays the Top-K passages side by side.

This makes the effect of domain fine-tuning directly visible.

> **Note:** Cosine similarity scores are model-specific. Rankings should be compared across different embedding models rather than directly comparing raw similarity scores.

---

## 🔌 API

The React frontend communicates with the Python backend through a FastAPI REST API.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Backend health check |
| `POST /api/chat` | Retrieve evidence and generate a grounded answer |
| `POST /api/retrieve` | Fine-Tuned MuRIL retrieval only |
| `POST /api/compare` | Base MuRIL vs Fine-Tuned MuRIL comparison |

### Example: `POST /api/chat`

**Request**

```json
{
  "question": "गेहूं को पानी कब दें?",
  "model_choice": "Gemini API"
}
```

**Response**

```json
{
  "answer": "...",
  "retrieved": [
    {
      "rank": 1,
      "score": 0.78,
      "chunk_id": "...",
      "document_id": "...",
      "title": "...",
      "text": "...",
      "source": "...",
      "url": "..."
    }
  ]
}
```

---

## 🏛️ Architecture

```text
React + Vite Frontend
        ↓
REST / JSON
        ↓
FastAPI Backend
        ↓
RAG Pipeline
        ↓
Fine-Tuned MuRIL + FAISS
        ↓
Retrieved Agriculture Context
        ↓
Gemini API / Local Qwen
        ↓
Grounded Hindi Answer
```

---

## 🛠️ Tech Stack

### AI / Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Google MuRIL
- FAISS

### LLM / Generation

- Google Gemini API
- Qwen through Ollama

### Backend

- FastAPI
- Uvicorn

### Frontend

- React
- Vite

### Data Processing

- Pandas
- NumPy
- PyMuPDF
- BeautifulSoup

### Training & Evaluation

- Hugging Face Datasets
- Sentence Transformers evaluation utilities
- scikit-learn
- FAISS

### Development Tools

- Git
- GitHub
- VS Code
- Kaggle
- Jupyter Notebook

### Legacy / Fallback Interface

- Gradio

---

## 📂 Project Structure

```text
rag-muril-finetuning/
│
├── api.py
├── app.py
├── requirements.txt
├── README.md
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       └── api/
│           └── client.js
│
├── rag/
│   ├── config.py
│   ├── retriever.py
│   ├── base_retriever.py
│   ├── pipeline.py
│   ├── llm.py
│   ├── pdf_loader.py
│   ├── build_faiss_index.py
│   └── build_base_faiss_index.py
│
├── data/
│   ├── chunks/
│   ├── cleaned/
│   ├── cleaned_v2/
│   ├── training/
│   ├── training_v2/
│   ├── validation/
│   ├── validation_v2/
│   └── index/
│       ├── finetuned.faiss
│       ├── base.faiss
│       └── finetuned_metadata.csv
│
├── models/
│   ├── base_muril/
│   ├── fine_tuned_muril_v2/
│   └── fine_tuned_muril_v3/
│
├── scripts/
│   ├── scraping/
│   ├── preprocessing/
│   ├── cleaning/
│   ├── question_generation/
│   ├── training/
│   │   ├── mine_hard_negatives_v2.py
│   │   ├── clean_hard_negatives_v2.py
│   │   └── train_muril_v3_hardneg_mnrl.py
│   ├── validation/
│   └── evaluation/
│
└── evaluation/
    ├── output/
    └── output_v2/
```

> Model weights are kept outside normal Git tracking because trained SentenceTransformer checkpoints are large.

---

## 🚀 Running Locally

### Prerequisites

- Python 3.11 recommended
- Node.js 18+
- npm
- Optional: Ollama for local Qwen inference
- Google Gemini API key if using Gemini

### 1. Clone the repository

```bash
git clone https://github.com/Prathamesh185/rag-muril-finetuning.git
cd rag-muril-finetuning
```

### 2. Create and activate a Python environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
```

Do not commit the `.env` file.

### 5. Start the FastAPI backend

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

---

## 🖥️ Local Qwen Setup

The project also supports local answer generation using Qwen through Ollama.

```bash
ollama pull qwen3.5:4b
```

Then start Ollama and select **Local Qwen** from the application.

Gemini can be used instead when a valid Google API key is configured.

---

## 🧪 Evaluation Outputs

The project includes evaluation results for:

- Base MuRIL
- Fine-Tuned MuRIL V2
- Fine-Tuned MuRIL V3
- E5-base
- BGE-M3

Key reported retrieval metrics include:

- Accuracy@1
- Accuracy@3
- Accuracy@5
- Accuracy@10
- Precision@K
- Recall@K
- MRR@10
- nDCG@10
- MAP@100

Fine-Tuned MuRIL V3 achieved the strongest retrieval performance among the evaluated models on the current Hindi agriculture test set.

---

## 🎯 Future Work

- Extend the system to Kannada and additional Indian languages
- Build an independent external agriculture retrieval benchmark
- Explore Triplet Loss and other contrastive objectives
- Study cross-lingual retrieval behavior
- Improve robustness to spelling variation and transliteration
- Release the trained encoder through Hugging Face
- Publicly deploy the complete application

---

## 📚 Data Source & Acknowledgments

- Hindi agriculture content sourced from **Vikaspedia**
- Encoder built on **MuRIL — Multilingual Representations for Indian Languages**
- Answer generation using **Google Gemini API**
- Local answer generation using **Qwen through Ollama**
- Built using **Sentence Transformers**
- Dense retrieval using **FAISS**

---

## 📄 License

This project is developed for academic and research purposes.
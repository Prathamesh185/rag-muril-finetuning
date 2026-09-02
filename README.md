# 🌾 Agriculture-Aware RAG using Fine-Tuned MuRIL

This project fine-tunes Google's **MuRIL model as an agriculture-aware sentence encoder** to improve domain-specific semantic retrieval for Indian-language Retrieval-Augmented Generation (RAG).

The current implementation focuses primarily on **Hindi agriculture data**. The system uses **Fine-Tuned MuRIL, FAISS, Gemini API / Local Qwen, FastAPI, and React** to retrieve relevant agriculture passages and generate grounded answers.

> **Core Focus:** Fine-tuning MuRIL for agriculture-specific semantic retrieval and evaluating its improvement over Base MuRIL.

---

## ✨ Highlights

* Agriculture-specific semantic retrieval
* Fine-tuned MuRIL sentence encoder
* Multiple Negatives Ranking Loss (MNRL)
* Hindi agriculture question–passage dataset
* FAISS-based dense retrieval
* Grounded RAG using Gemini API or Local Qwen
* Live Base MuRIL vs Fine-Tuned MuRIL comparison
* Retrieval Analysis with real passages, scores, sources, and URLs
* React + FastAPI web application
* Dataset validation for duplicates and leakage
* Designed for extension to additional Indian languages

---

## ✅ Project Status

**Completed:**

- [x] Hindi agriculture data collected and cleaned (Vikaspedia)
- [x] Sentence-based chunking and question generation pipeline
- [x] Question–passage dataset built and validated (duplicates, leakage) — **20,141 pairs**, 7,379 unique chunks, 1,814 documents
- [x] MuRIL fine-tuned using MNRL (V2) — see [Fine-Tuning Configuration](#️-fine-tuning-configuration)
- [x] Base vs Fine-Tuned MuRIL evaluated on held-out V2 test set (metrics below)
- [x] FAISS indexes built for both Base and Fine-Tuned encoders
- [x] RAG pipeline wired to Gemini API and Local Qwen (via Ollama)
- [x] FastAPI backend with 4 endpoints (`/api/health`, `/api/chat`, `/api/retrieve`, `/api/compare`)
- [x] React + Vite frontend with 3 interfaces: AI Assistant, Retrieval Analysis, Model Comparison
- [x] Legacy Gradio interface (fallback)

**In progress / not yet done:**

- [ ] Hard-negative mining for a further fine-tuning pass
- [ ] Triplet-loss experiment (currently MNRL only)
- [ ] Comparison against newer multilingual encoders (E5, BGE)
- [ ] Support for additional Indian languages (e.g. Kannada)
- [ ] Public deployment
- [ ] Hugging Face model release

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
MuRIL Fine-Tuning using MNRL
        ↓
Fine-Tuned MuRIL Encoder
```

### Runtime RAG Pipeline

```text
User Question
      ↓
Fine-Tuned MuRIL V2
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

### Base vs Fine-Tuned Retrieval Comparison

```text
                    ┌── Base MuRIL ───────→ Base FAISS ───────┐
User Question ──────┤                                          ├──→ Compare Top-K Results
                    └── Fine-Tuned MuRIL ─→ Fine-Tuned FAISS ─┘
```

Both retrieval paths use the same passage corpus and metadata ordering so that the encoder is the main variable being compared.

---

## 📊 Evaluation

The project evaluates:

* **Base MuRIL**
* **Fine-Tuned MuRIL V2**

Both models are evaluated under the same retrieval setup; only the sentence encoder changes.

### Final V2 Retrieval Results

| Metric     | Base MuRIL | Fine-Tuned MuRIL |
| ---------- | ---------: | ---------------: |
| Accuracy@1 |     21.46% |       **70.10%** |
| Recall@5   |     39.39% |       **93.18%** |
| Recall@10  |     48.84% |       **96.62%** |
| MRR@10     |     0.2919 |       **0.7999** |
| nDCG@10    |     0.3383 |       **0.8410** |
| MAP@100    |     0.3063 |       **0.8013** |

### Improvement

* **Accuracy@1:** +48.64 percentage points
* **Recall@5:** +53.79 percentage points
* **Recall@10:** +47.78 percentage points
* **MRR@10:** +0.5081
* **nDCG@10:** +0.5027
* **MAP@100:** +0.4950

The results show that domain-specific fine-tuning substantially improves MuRIL's ability to retrieve agriculture passages for Hindi agriculture queries.

> The offline V2 evaluation corpus and the full application FAISS index serve different purposes. Evaluation is performed on the held-out test setup, while the live application retrieves from the broader indexed agriculture passage corpus.

---

## ⚙️ Fine-Tuning Configuration

| Setting             | Value                                  |
| ------------------- | --------------------------------------- |
| Base model          | `google/muril-base-cased`              |
| Training objective  | Multiple Negatives Ranking Loss (MNRL) |
| Epochs              | 3                                      |
| Batch size          | 32                                     |
| Learning rate       | `2e-5`                                 |
| Max sequence length | 256                                    |
| Embedding dimension | 768                                    |
| Training pairs      | 16,292                                 |
| Validation pairs    | 1,869                                  |

---

## 🧹 Dataset Validation

Before final training and evaluation, the dataset was checked for:

* Exact duplicate documents
* Near-duplicate documents
* Exact duplicate questions
* Near-duplicate questions
* Question-to-passage copying
* Train / validation / test leakage
* Duplicate-document grouping across splits

### Final V2 Dataset

* **20,141** question–passage pairs
* **7,379** unique chunks
* **1,814** documents
* **1,770** split groups

---

## 🧠 Why MuRIL?

MuRIL was selected because it was specifically developed for Indian languages and supports multilingual and transliterated Indian-language text.

The goal of this project is not only to use MuRIL directly, but to adapt it to the **agriculture domain** so that semantically relevant agriculture passages are ranked higher for user questions.

Fine-tuning converts the general-purpose MuRIL encoder into a more domain-aware retrieval model.

---

## 🌐 Web Application

The final application contains three main interfaces.

### 1. AI Assistant

Users can ask agriculture questions in Hindi and receive:

* Grounded answers
* Retrieved evidence passages
* Similarity scores
* Source names
* Source URLs

Users can choose between:

* **Gemini API**
* **Local Qwen through Ollama**

### 2. Retrieval Analysis

Shows the real retrieval pipeline:

```text
User Query
→ Fine-Tuned MuRIL
→ Query Embedding
→ FAISS
→ Top-K Passages
```

It displays the actual retrieved agriculture passages before LLM generation.

### 3. Model Comparison

Runs the same question through:

* Base MuRIL
* Fine-Tuned MuRIL V2

and displays the Top-K passages side by side.

This makes the effect of domain fine-tuning directly visible.

---

## 🔌 API

The React frontend communicates with the Python backend through a FastAPI REST API.

| Endpoint             | Purpose                                          |
| --------------------- | ------------------------------------------------ |
| `GET /api/health`    | Backend health check                             |
| `POST /api/chat`     | Retrieve evidence and generate a grounded answer |
| `POST /api/retrieve` | Fine-Tuned MuRIL retrieval only                  |
| `POST /api/compare`  | Base MuRIL vs Fine-Tuned MuRIL comparison        |

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

### Architecture

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
Gemini API / Local Qwen
```

---

## 🛠️ Tech Stack

### AI / Machine Learning

* Python
* PyTorch
* Hugging Face Transformers
* Sentence Transformers
* Google MuRIL
* FAISS

### LLM / Generation

* Google Gemini API
* Qwen through Ollama

### Backend

* FastAPI
* Uvicorn

### Frontend

* React
* Vite

### Data Processing

* Pandas
* NumPy
* PyMuPDF
* BeautifulSoup

### Training & Evaluation

* Hugging Face Datasets
* Sentence Transformers evaluation utilities
* scikit-learn
* FAISS

### Development Tools

* Git
* GitHub
* VS Code
* Kaggle
* Jupyter Notebook

### Legacy / Fallback Interface

* Gradio

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
│   └── fine_tuned_muril_v2/
│
├── scripts/
│   ├── scraping/
│   ├── preprocessing/
│   ├── cleaning/
│   ├── question_generation/
│   ├── training/
│   ├── validation/
│   └── evaluation/
│
└── evaluation/
    ├── output/
    └── output_v2/
```

---

## 🚀 Running Locally

### Prerequisites

* Python 3.11 recommended
* Node.js 18+ and npm
* (Optional) [Ollama](https://ollama.com) for local Qwen inference
* A Google Gemini API key (if using Gemini instead of local Qwen)

### 1. Clone the repository

```bash
git clone https://github.com/Prathamesh185/rag-muril-finetuning.git
cd rag-muril-finetuning
```

### 2. Create and activate a Python environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

Do not commit the `.env` file.

### 5. Start the FastAPI backend

From the project root:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

### 6. Start the React frontend

Open another terminal:

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

Install Ollama and ensure the required model is available:

```bash
ollama pull qwen3.5:4b
```

Then start Ollama and select **Local Qwen** from the application.

Gemini can be used instead when a valid Google API key is configured.

---

## 🧪 Evaluation Outputs

Stored V2 evaluation outputs are available under:

```text
evaluation/output_v2/
```

Important files include:

```text
base_results.json
finetuned_results.json
model_comparison.csv
retrieval_examples.csv
retrieval_examples_sorted.csv
best_demo_examples.csv
```

These contain the quantitative and qualitative results used to compare Base MuRIL with Fine-Tuned MuRIL V2.

---

## 🎯 Future Work

* Extend the system to additional Indian languages such as Kannada
* Explore hard-negative mining
* Experiment with Triplet Loss and other contrastive objectives
* Compare with newer multilingual embedding models such as E5 and BGE
* Expand independent agriculture evaluation datasets
* Improve multilingual retrieval robustness
* Release the trained encoder through Hugging Face
* Deploy the full application publicly

---

## 📚 Data Source & Acknowledgments

* Hindi agriculture content sourced from [Vikaspedia](https://vikaspedia.in)
* Encoder built on **MuRIL** (Multilingual Representations for Indian Languages) — Google Research
* Answer generation via **Google Gemini API** and **Qwen** (via Ollama)
* Built with the **Sentence Transformers** and **FAISS** open-source libraries

---

## 📄 License

This project is developed for academic and research purposes.


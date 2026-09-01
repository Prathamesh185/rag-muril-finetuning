# 🌾 Domain-Specific Multilingual RAG using Fine-Tuned MuRIL

This project fine-tunes Google’s **MuRIL model as an agriculture-aware sentence encoder** to improve domain-specific semantic retrieval for Indian-language RAG systems.

The current implementation focuses primarily on **Hindi agriculture data**, while the architecture is designed to support additional Indian languages.

The system combines **Fine-Tuned MuRIL, FAISS, and Gemini / Local Qwen** to retrieve relevant agriculture passages and generate grounded answers.

> **Core Focus:** Fine-tuning MuRIL for agriculture-specific semantic retrieval and evaluating its improvement over Base MuRIL.

---

## ✨ Highlights

- Agriculture-specific semantic retrieval
- Fine-tuned MuRIL sentence encoder
- Multiple Negatives Ranking Loss (MNRL)
- Hindi agriculture question–passage dataset
- FAISS-based dense retrieval
- Grounded RAG using Gemini / Local Qwen
- Base MuRIL vs Fine-Tuned MuRIL evaluation
- Designed for extension to additional Indian languages

---

## 🏗️ Pipeline

```text
Vikaspedia Hindi Agriculture Articles
        ↓
Data Collection & Cleaning
        ↓
Sentence-Based Chunking
        ↓
Question Generation
        ↓
Question–Passage Dataset
        ↓
Dataset Validation & Splitting
        ↓
MuRIL Fine-Tuning using MNRL
        ↓
FAISS Index
        ↓
Semantic Retrieval
        ↓
Gemini / Local Qwen
        ↓
Grounded RAG Answer
```

## 📊 Evaluation

The project compares retrieval performance between:

- **Base MuRIL**
- **Fine-Tuned MuRIL**

Both models are evaluated using the same held-out V2 test set and the same unique-passage retrieval corpus.

### Test Setup

- Test questions: **1,980**
- Unique passages: **744**
- Similarity function: **Cosine Similarity**
- Retrieval depth: **Top-10**
- Evaluation corpus built using unique `chunk_id` values

### Final Retrieval Results

| Metric | Base MuRIL | Fine-Tuned MuRIL |
|---|---:|---:|
| Accuracy@1 | 21.46% | **70.10%** |
| Accuracy@5 | 39.39% | **93.18%** |
| Recall@10 | 48.84% | **96.62%** |
| MRR@10 | 0.2919 | **0.7999** |
| nDCG@10 | 0.3383 | **0.8410** |
| MAP@100 | 0.3063 | **0.8013** |

### Improvement

- **Accuracy@1:** +48.64 percentage points
- **Recall@10:** +47.78 percentage points
- **MRR@10:** +0.5081
- **nDCG@10:** +0.5027
- **MAP@100:** +0.4950

The fine-tuned MuRIL model shows a substantial improvement over Base MuRIL for Hindi agriculture semantic retrieval.

---

## ⚙️ Fine-Tuning Configuration

| Setting | Value |
|---|---|
| Base model | `google/muril-base-cased` |
| Training objective | Multiple Negatives Ranking Loss (MNRL) |
| Epochs | 3 |
| Batch size | 32 |
| Learning rate | `2e-5` |
| Max sequence length | 256 |
| Embedding dimension | 768 |
| Training pairs | 16,292 |
| Validation pairs | 1,869 |
---

## 🧹 Dataset Validation

Before training, the dataset was checked for:

- exact and near-duplicate documents
- duplicate and near-duplicate questions
- question-to-passage copying
- train/validation/test leakage
- duplicate-document grouping across splits

Final V2 dataset:

- **20,141** question-passage pairs
- **7,379** unique chunks
- **1,814** documents
- **1,770** split groups
---

## 🛠️ Tech Stack

### AI / Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Google MuRIL
- FAISS

### LLM

- Google Gemini API

### Frameworks

- LangChain
- Gradio

### Data Processing

- Pandas
- NumPy

### Development Tools

- Git
- GitHub
- VS Code
- Jupyter Notebook
- Kaggle

---

## 📂 Project Structure

```text
project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── chunks/
│   ├── generated_questions/
│   └── training/
│
├── scripts/
│   ├── preprocessing/
│   ├── dataset/
│   ├── training/
│   └── evaluation/
│
├── models/
├── evaluation/
├── logs/
└── outputs/
```

---

## 🎯 Future Work

- Support additional Indian languages
- Improve hard negative mining
- Fine-tune using Triplet Loss
- Compare with newer multilingual embedding models
- Release the trained model on Hugging Face
- Deploy the application publicly

---

## 📄 License

This project is developed for academic and research purposes.

---

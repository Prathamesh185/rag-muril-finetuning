# 🌾 Domain-Specific Multilingual RAG using Fine-Tuned MuRIL

Fine-tunes Google’s **MuRIL sentence encoder** for agriculture-specific semantic retrieval in Indian languages and integrates the improved encoder into a multilingual **Retrieval-Augmented Generation (RAG)** system.

The project presents an end-to-end pipeline covering **agriculture data collection, dataset generation, MuRIL fine-tuning, retrieval evaluation, FAISS-based semantic search, and RAG-powered question answering**.

> **Core Focus:** The main technical contribution of this project is the **fine-tuned agriculture-aware MuRIL sentence encoder**. The RAG application demonstrates how the improved encoder can be used for real-world agriculture information retrieval and question answering.

---

## 🚀 Overview

Large Language Models (LLMs) do not always have reliable access to specialized, private, or up-to-date agriculture knowledge. **Retrieval-Augmented Generation (RAG)** addresses this limitation by retrieving relevant information from an external knowledge base before generating an answer.

However, the effectiveness of a RAG system depends heavily on the quality of its **retrieval component**. If the wrong passage is retrieved, even a powerful LLM may generate an inaccurate or irrelevant answer.

General-purpose multilingual models can understand Indian languages, but they are not specifically optimized for **agriculture-related semantic retrieval**.

This project fine-tunes **Google MuRIL (Multilingual Representations for Indian Languages)** as a sentence encoder using agriculture-specific question–passage pairs. The goal is to adapt its embeddings so that relevant agriculture questions and passages are represented closer together in the embedding space.

**Hindi is currently the primary language used for model training and evaluation**, while the architecture is designed to support additional Indian languages.

The fine-tuned MuRIL encoder is evaluated against **Base MuRIL** using standard Information Retrieval (IR) metrics. The improved encoder is then integrated with **FAISS and an LLM** to demonstrate its use in an end-to-end agriculture RAG system.

---

## ✨ Highlights

- 🌾 Agriculture-specific semantic retrieval
- 🤖 Fine-tuned MuRIL sentence encoder
- 🧠 Multiple Negatives Ranking Loss (MNRL)
- 📚 Agriculture question–passage dataset
- 🔎 FAISS vector similarity search
- ⛓️ LangChain-based RAG pipeline
- 💬 LLM-based grounded answer generation
- 🖥️ Gradio web application
- 📊 Base MuRIL vs Fine-Tuned MuRIL evaluation
- 🌐 Extensible to additional Indian languages

---

## 🧠 Core Concepts

| Term | Simple Meaning | Example / Role in This Project |
| --- | --- | --- |
| **Sentence Encoder** | Converts a sentence or passage into numbers called an **embedding (vector)** so that texts with similar meanings can be matched. | `"गेहूं को पानी कब दें?"` → Embedding vector |
| **MuRIL** | Google's multilingual model designed for Indian languages. We use it as our **Sentence Encoder**. | MuRIL → Agriculture sentence embeddings |
| **Fine-Tuning** | Taking an already-trained model and training it further on specific data to adapt it to a particular domain or task. | Base MuRIL → Agriculture-Aware MuRIL |
| **MNRL** | A training method that teaches the encoder to bring a question closer to its **relevant passage** and farther from unrelated passages. | Question ↔ Relevant Passage |
| **Semantic Retrieval** | Finds relevant information based on the **meaning of the query**, rather than only matching exact words. | `"गेहूं को पानी कब दें?"` ↔ `"पहली सिंचाई 20–25 दिन बाद..."` |
| **FAISS** | Searches through stored embeddings to quickly find the passages most similar to the user's question. | Query Embedding → Top-K Relevant Passages |
| **RAG** | First retrieves relevant information from trusted documents and then gives it to an LLM as context for generating the answer. | Question → Retrieval → Context → LLM → Answer |

> **Project Focus:** The main technical contribution is the **fine-tuned agriculture-aware MuRIL sentence encoder**. The RAG system demonstrates how the improved encoder can be used for agriculture question answering.

---

## 🏗️ Project Pipeline

```text
Government Agriculture Documents
                │
                ▼
      Text Extraction
                │
                ▼
      Text Cleaning
                │
                ▼
     Semantic Chunking
                │
                ▼
     Question Generation
                │
                ▼
 Question–Passage Pairs
                │
                ▼
     MuRIL Fine-Tuning
                │
                ▼
 Improved Sentence Embeddings
                │
                ▼
        FAISS Index
                │
                ▼
     Semantic Retrieval
                │
                ▼
 Retrieval-Augmented Generation
                │
                ▼
     Gradio Web Application
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

## 🌟 Contributions

- Fine-tuned MuRIL for agriculture-specific semantic retrieval.
- Built a multilingual agriculture retrieval dataset.
- Improved document retrieval for agriculture RAG systems.
- Evaluated retrieval performance using standard IR metrics.
- Developed an end-to-end multilingual agriculture question-answering pipeline.

---

## 🚜 Applications

The fine-tuned encoder can be integrated into:

- Government agriculture portals
- ICAR knowledge systems
- Farmer assistance chatbots
- Agriculture helplines
- University agriculture knowledge bases
- NGOs
- Private agriculture companies
- Agriculture RAG applications

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

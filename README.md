# 🌾 Agriculture-Aware Multilingual RAG using Fine-Tuned MuRIL

A research project that improves multilingual agricultural **Retrieval-Augmented Generation (RAG)** by fine-tuning the **MuRIL** sentence encoder to retrieve more relevant agricultural information in Indian languages. The project builds an end-to-end pipeline covering data collection, dataset generation, encoder fine-tuning, retrieval evaluation, and RAG deployment.

---

## 🚀 Overview

General-purpose multilingual embedding models often struggle with domain-specific retrieval tasks. This project addresses that challenge by fine-tuning **MuRIL** on Hindi agriculture question–passage pairs generated from government agricultural resources.

The fine-tuned encoder is integrated into a complete **Retrieval-Augmented Generation (RAG)** pipeline to improve retrieval quality for agriculture-related queries in Indian languages.

---

## ✨ Highlights

- 🌾 Agriculture-specific multilingual retrieval
- 🤖 Fine-Tuned MuRIL sentence encoder
- 🧠 Multiple Negatives Ranking Loss (MNRL)
- 🎯 Hard Negative Mining
- 🔥 Triplet Loss fine-tuning
- 📚 FAISS semantic vector search
- ⛓️ LangChain-based RAG pipeline
- 💬 Gradio web application
- 📊 Retrieval evaluation using standard IR metrics

---

## 📌 Features

- Fine-tuned MuRIL sentence embeddings
- Hindi agriculture retrieval dataset
- Question generation pipeline
- Hard negative mining
- Triplet dataset creation
- FAISS vector database
- Semantic document retrieval
- LangChain RAG pipeline
- Google Gemini integration
- Interactive Gradio interface
- Retrieval performance evaluation

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
     Hard Negative Mining
                │
                ▼
 Question–Passage Triplets
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
    Retrieval-Augmented
        Generation (RAG)
                │
                ▼
     Gradio Web Interface
```

---

## 🛠️ Tech Stack

### AI / Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- MuRIL
- FAISS
- LangChain
- Gradio

### Data Processing

- Pandas
- NumPy

### LLM

- Google Gemini API

### Development Tools

- Git
- GitHub
- VS Code
- Google Colab
- Jupyter Notebook

---

## 📂 Project Structure

```text
project/
│
├── app.py
├── config.py
├── data.py
├── llm.py
├── pdf_loader.py
├── retriever.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── chunks/
│   ├── generated_questions/
│   └── training/
│
├── scripts/
│
├── models/
│
├── evaluation/
│
└── README.md
```

---

## 📊 Evaluation

The retrieval system is evaluated using standard Information Retrieval (IR) metrics.

### Metrics

- Recall@1
- Recall@5
- Recall@10
- MRR@10
- nDCG@10
- MAP@10

### Model Comparison

The project compares retrieval performance of:

- Base MuRIL
- Fine-Tuned MuRIL
- Triplet Fine-Tuned MuRIL

using the same held-out test dataset.

> **Note:** Final evaluation scores and comparison tables will be added after completion of experiments.

---

## 📁 Dataset

The training dataset is constructed from publicly available government agriculture resources.

Pipeline:

- Government agriculture documents
- Text extraction
- Cleaning and preprocessing
- Semantic chunking
- Automatic question generation
- Hard negative mining
- Question–passage triplet creation

The resulting dataset is used to fine-tune MuRIL for agriculture-specific semantic retrieval.

---

## 🔬 Research Contribution

This project contributes to multilingual agricultural retrieval by:

- Fine-tuning MuRIL on agriculture question–passage pairs
- Improving semantic retrieval for Retrieval-Augmented Generation
- Comparing retrieval performance before and after fine-tuning
- Building an end-to-end multilingual RAG pipeline
- Evaluating retrieval quality using standard IR metrics

---

## 🎯 Future Work

- Support additional Indian languages
- Expand the agriculture corpus
- Explore improved hard negative mining techniques
- Compare with newer multilingual embedding models
- Release trained embeddings on Hugging Face
- Deploy the application for public use

---

## 📄 License

This project is developed for academic and research purposes.

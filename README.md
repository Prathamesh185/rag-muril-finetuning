# 🌾 Domain-Specific Multilingual RAG using Fine-Tuned MuRIL

A research project that improves multilingual agricultural **Retrieval-Augmented Generation (RAG)** by fine-tuning Google's **MuRIL** sentence encoder for agriculture-specific semantic retrieval in Indian languages.

The project presents an end-to-end pipeline covering **data collection, dataset generation, sentence encoder fine-tuning, retrieval evaluation, and RAG deployment.**

---

## 🚀 Overview

Large Language Models (LLMs) such as ChatGPT do not automatically have access to domain-specific knowledge sources, including agricultural manuals, reports, and advisory documents published by organizations such as **ICAR (Indian Council of Agricultural Research)**. As a result, they may generate less relevant responses to specialized agriculture-related queries.

**Retrieval-Augmented Generation (RAG)** addresses this limitation by retrieving relevant documents before generating an answer. However, the effectiveness of a RAG system depends heavily on the quality of its retrieval component.

This project fine-tunes **MuRIL**, a multilingual sentence encoder for Indian languages, on agriculture-specific question–passage pairs constructed from publicly available government agricultural documents. The fine-tuned encoder is then integrated into an end-to-end multilingual RAG pipeline to improve semantic retrieval for agriculture-related question answering.

---

## ✨ Highlights

- 🌾 Agriculture-specific multilingual retrieval
- 🤖 Fine-tuned MuRIL sentence encoder
- 🧠 Multiple Negatives Ranking Loss (MNRL)
- 🎯 Hard Negative Mining
- 🔥 Triplet Loss fine-tuning
- 📚 FAISS semantic vector search
- ⛓️ LangChain-based RAG pipeline
- 💬 Google Gemini integration
- 🖥️ Gradio web application
- 📊 Retrieval evaluation using standard Information Retrieval (IR) metrics

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
 Retrieval-Augmented Generation
                │
                ▼
     Gradio Web Application
```

---

## 📊 Evaluation

The project compares retrieval performance between:

- **Base MuRIL**
- **Fine-Tuned MuRIL**

using identical evaluation settings and the same held-out test dataset.

### Evaluation Metrics

- Accuracy@1
- Accuracy@3
- Accuracy@5
- Accuracy@10
- Recall@1
- Recall@3
- Recall@5
- Recall@10
- Precision@1
- Precision@3
- Precision@5
- Precision@10
- MRR@10
- nDCG@10
- MAP@10

> Final evaluation results and comparison tables will be added after all experiments are completed.

| Metric | Base MuRIL | Fine-Tuned MuRIL |
|---------|------------|------------------|
| Accuracy@1 | - | - |
| Accuracy@5 | - | - |
| Recall@10 | - | - |
| MRR@10 | - | - |
| nDCG@10 | - | - |
| MAP@10 | - | - |

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

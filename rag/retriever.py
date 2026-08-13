# import numpy as np
# from sentence_transformers import util

# from rag.config import encoder
# from rag.data import base_documents
# from rag.llm import gemini_llm, local_llm
# from rag import pdf_loader

# # Pre-compute embeddings for base knowledge
# base_embeddings = encoder.encode(
#     base_documents,
#     normalize_embeddings=True
# )

# print(f"Base documents loaded: {len(base_documents)}")


# def answer(question, model_choice):
#     """
#     Main RAG pipeline.
#     """

#     # -----------------------------
#     # Combine documents
#     # -----------------------------
#     all_documents = list(base_documents)

#     if pdf_loader.pdf_documents:
#         all_documents.extend(pdf_loader.pdf_documents)

#     # -----------------------------
#     # Combine embeddings
#     # -----------------------------
#     if pdf_loader.pdf_embeddings is not None:

#         all_embeddings = np.vstack([
#             base_embeddings,
#             pdf_loader.pdf_embeddings
#         ])

#     else:

#         all_embeddings = base_embeddings

#     # -----------------------------
#     # Encode question
#     # -----------------------------
#     question_embedding = encoder.encode(
#         question,
#         normalize_embeddings=True
#     )

#     # -----------------------------
#     # Similarity Search
#     # -----------------------------
#     similarity_scores = util.cos_sim(
#         question_embedding,
#         all_embeddings
#     )[0]

#     top_results = similarity_scores.topk(5)

#     context_chunks = []
#     source_list = []

#     print("\n========== Retrieved Chunks ==========")

#     for index, score in zip(
#         top_results.indices,
#         top_results.values
#     ):

#         index = int(index)
#         score = float(score)

#         source = (
#             "PDF"
#             if index >= len(base_documents)
#             else "Base Knowledge"
#         )

#         print(f"\nSource : {source}")
#         print(f"Score  : {score:.4f}")
#         print(all_documents[index])

#         if score > 0.30:

#             context_chunks.append(all_documents[index])

#             source_list.append(
#                 f"{source} (Score: {score:.2f})"
#             )

#     print("======================================\n")

#     # -----------------------------
#     # No context found
#     # -----------------------------
#     if not context_chunks:

#         return "इस विषय पर जानकारी उपलब्ध नहीं है।"

#     # -----------------------------
#     # Build Context
#     # -----------------------------
#     context = "\n".join(context_chunks)
#     context = context[:3000]

#     print(f"Using {len(context_chunks)} retrieved passages")

#     # -----------------------------
#     # Prompt
#     # -----------------------------
#     prompt = f"""
# आप एक कृषि सहायक हैं।

# केवल नीचे दिए गए संदर्भ के आधार पर उत्तर दें।

# नियम:
# 1. उत्तर केवल संदर्भ से दें।
# 2. बाहरी जानकारी का उपयोग न करें।
# 3. यदि उत्तर संदर्भ में नहीं है तो लिखें:
#    "इस विषय पर जानकारी उपलब्ध नहीं है।"
# 4. उत्तर सरल हिन्दी में दें।
# 5. सभी तथ्य और संख्याएँ सही रखें।

# संदर्भ:
# {context}

# प्रश्न:
# {question}

# उत्तर:
# """

#     # -----------------------------
#     # Generate Answer
#     # -----------------------------
#     try:

#         if model_choice == "Gemini API":
#             answer_text = gemini_llm(prompt)
#         else:
#             answer_text = local_llm(prompt)

#     except Exception as e:

#         print(f"Generation Error: {e}")

#         answer_text = local_llm(prompt)

#     # -----------------------------
#     # Add Sources
#     # -----------------------------
#     sources_text = "\n\nSources:\n"

#     for source in source_list:
#         sources_text += f"• {source}\n"

#     return answer_text + sources_text


import faiss
import pandas as pd

from rag.config import encoder
from rag.llm import gemini_llm, local_llm


# ==========================================================
# CONFIG
# ==========================================================

INDEX_FILE = "data/index/finetuned.faiss"

METADATA_FILE = "data/index/finetuned_metadata.csv"


# ==========================================================
# LOAD INDEX + METADATA
# ==========================================================

print("Loading FAISS index...")

index = faiss.read_index(
    INDEX_FILE
)

metadata = pd.read_csv(
    METADATA_FILE,
    encoding="utf-8-sig"
)

print(f"FAISS vectors loaded : {index.ntotal:,}")
print(f"Metadata rows        : {len(metadata):,}")

assert index.ntotal == len(metadata)


# ==========================================================
# RETRIEVAL
# ==========================================================

def retrieve(question, top_k=5):

    query_embedding = encoder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (idx, score) in enumerate(
        zip(
            indices[0],
            scores[0]
        ),
        start=1
    ):

        row = metadata.iloc[int(idx)]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "text": row["chunk_text"],
                "source": row["source"],
                "url": row["url"]
            }
        )

    return results

def answer(question, model_choice):

    # ======================================================
    # RETRIEVE TOP PASSAGES
    # ======================================================

    results = retrieve(
        question,
        top_k=5
    )

    if not results:
        return "इस विषय पर जानकारी उपलब्ध नहीं है।"


    # ======================================================
    # BUILD CONTEXT
    # ======================================================

    context_blocks = []

    for result in results:

        block = f"""
स्रोत {result["rank"]}
शीर्षक: {result["title"]}

{result["text"]}
"""

        context_blocks.append(
            block.strip()
        )

    context = "\n\n".join(
        context_blocks
    )


    # ======================================================
    # BUILD PROMPT
    # ======================================================

    prompt = f"""
आप एक कृषि सूचना सहायक हैं।

केवल नीचे दिए गए स्रोतों के आधार पर प्रश्न का उत्तर दें।

नियम:
1. उत्तर केवल दिए गए संदर्भ से दें।
2. बाहरी जानकारी का उपयोग न करें।
3. यदि पर्याप्त जानकारी उपलब्ध नहीं है तो लिखें:
   "उपलब्ध स्रोतों में इस प्रश्न का पर्याप्त उत्तर नहीं मिला।"
4. उत्तर सरल और स्पष्ट हिन्दी में दें।
5. संख्याएँ और कृषि संबंधी तथ्य सही रखें।
6. उत्तर छोटा और उपयोगी रखें।

संदर्भ:

{context}

प्रश्न:
{question}

उत्तर:
"""


    # ======================================================
    # GENERATE ANSWER
    # ======================================================

    try:

        if model_choice == "Gemini API":

            answer_text = gemini_llm(
                prompt
            )

        else:

            answer_text = local_llm(
                prompt
            )

    except Exception as e:

        print(
            "Generation Error:",
            e
        )

        answer_text = local_llm(
            prompt
        )


    # ======================================================
    # ADD SOURCES
    # ======================================================

    sources = "\n\nSources:\n"

    for result in results[:3]:

        sources += (
            f'\n{result["rank"]}. '
            f'{result["title"]}\n'
            f'{result["url"]}\n'
        )


    return answer_text + sources
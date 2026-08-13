from rag.retriever import retrieve
from rag.llm import gemini_llm, local_llm


# ==========================================================
# BUILD CONTEXT
# ==========================================================

def build_context(results):

    blocks = []

    for result in results:

        block = f"""
स्रोत {result["rank"]}
शीर्षक: {result["title"]}

जानकारी:
{result["text"]}
"""

        blocks.append(
            block.strip()
        )

    return "\n\n".join(blocks)


# ==========================================================
# BUILD PROMPT
# ==========================================================

def build_prompt(question, context):

    return f"""
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


# ==========================================================
# MAIN RAG PIPELINE
# ==========================================================

def rag_answer(question, model_choice):

    # Retrieve top passages
    results = retrieve(
        question,
        top_k=5
    )

    if not results:
        return {
            "answer": "इस विषय पर जानकारी उपलब्ध नहीं है।",
            "retrieved": []
        }

    # Build context
    context = build_context(results)

    # Build prompt
    prompt = build_prompt(
        question,
        context
    )

    # Generate answer
    try:

        if model_choice == "Gemini API":
            answer_text = gemini_llm(prompt)

        else:
            answer_text = local_llm(prompt)

    except Exception as e:

        print("Generation Error:", e)

        answer_text = local_llm(prompt)

    return {
        "answer": answer_text,
        "retrieved": results
    }
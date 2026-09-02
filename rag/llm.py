import ollama

from rag.config import client


def local_llm(prompt):

    response = ollama.chat(
        model="qwen3.5:4b",
        think=False,
        options={
            "num_predict": 300,
            "temperature": 0.1,
            "num_thread": 6,
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "आप एक कृषि सूचना सहायक हैं। "
                    "केवल उपयोगकर्ता द्वारा दिए गए संदर्भ के आधार पर उत्तर दें। "
                    "संदर्भ में जो जानकारी नहीं है उसे न जोड़ें। "
                    "संदर्भ में दिए गए महत्वपूर्ण तथ्य, संख्या, दिन, अवस्था या मात्रा को न छोड़ें। "
                    "उत्तर सीधे, सरल और स्पष्ट हिन्दी में दें। "
                    "अनावश्यक भूमिका या अतिरिक्त जानकारी न दें।"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    raw = response["message"]["content"]

    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    if raw.strip():
        return raw

    return "इस विषय पर जानकारी उपलब्ध नहीं है."


def gemini_llm(prompt):

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    return response.text

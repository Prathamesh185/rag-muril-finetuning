import ollama

from rag.config import client


SYSTEM_PROMPT = """
आप AgriSahayak हैं — एक कृषि सूचना सहायक।

आपका काम केवल दिए गए संदर्भ के आधार पर उपयोगकर्ता के प्रश्न का उत्तर देना है।

नियम:
1. केवल संदर्भ में उपलब्ध जानकारी का उपयोग करें।
2. संदर्भ में न दी गई जानकारी, अनुमान, सामान्य ज्ञान या अतिरिक्त सलाह न जोड़ें।
3. यदि संदर्भ में प्रश्न का उत्तर पर्याप्त रूप से उपलब्ध नहीं है, तो स्पष्ट रूप से कहें:
   "दिए गए संदर्भ में इस प्रश्न का पर्याप्त उत्तर उपलब्ध नहीं है।"
4. उत्तर सरल, स्पष्ट और स्वाभाविक हिन्दी में दें।
5. उपयोगकर्ता के प्रश्न का सीधा उत्तर पहले दें।
6. महत्वपूर्ण तथ्य न छोड़ें, विशेष रूप से:
   - समय / दिन
   - मात्रा
   - प्रतिशत
   - फसल की अवस्था
   - उपचार
   - सिंचाई
   - उर्वरक / पोषक तत्व
   - कीट / रोग नियंत्रण
7. हर उत्तर में कम से कम 1 और अधिकतम 3 सबसे महत्वपूर्ण factual phrases को
   **double asterisks** में अवश्य लिखें।
8. महत्वपूर्ण phrases को **double asterisks** में लिखते समय,
   जहाँ संभव हो उन्हें दिए गए संदर्भ से ठीक उसी wording में चुनें,
   ताकि वही तथ्य retrieved evidence में भी पहचाना जा सके।
9. मुख्य उत्तर का सबसे जरूरी भाग हमेशा highlight करें, जैसे:
   - समय
   - दिन / अवधि
   - मात्रा
   - प्रतिशत
   - फसल की अवस्था
   - रोग / कीट
   - उपचार
   - सिंचाई का समय
   - उर्वरक की मात्रा
10. केवल छोटे factual phrases को highlight करें।
    पूरे वाक्य या पूरे paragraph को bold न करें।
11. उदाहरण:
    "सर्पगंधा की जड़ को उखाड़ने का सबसे सही समय **जाड़े का मौसम** है,
    जब **पत्तियाँ झड़ जाती हैं**। इस समय जड़ों में
    **एल्केलाईड की मात्रा सबसे अधिक** होती है।"
12. किसी तथ्य में अनिश्चितता हो तो उसे निश्चित तथ्य की तरह न लिखें।
13. Markdown headings की आवश्यकता नहीं है। उत्तर साफ और compact रखें।
14. "संदर्भ के अनुसार", "उपलब्ध जानकारी के अनुसार" जैसी भूमिका बार-बार न लिखें।
"""

def local_llm(prompt):

    response = ollama.chat(
        model="qwen2.5:1.5b",
        think=False,
        options={
            "num_predict": 300,
            "temperature": 0.1,
            "num_thread": 6,
        },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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

    return "इस विषय पर जानकारी उपलब्ध नहीं है।"


def gemini_llm(prompt):

    full_prompt = f"""
{SYSTEM_PROMPT}

उपयोगकर्ता का प्रश्न और संदर्भ नीचे दिया गया है:

{prompt}

अब ऊपर दिए गए नियमों का पालन करते हुए उत्तर दें।
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=full_prompt,
    )

    if response.text and response.text.strip():
        return response.text.strip()

    return "इस विषय पर जानकारी उपलब्ध नहीं है।"
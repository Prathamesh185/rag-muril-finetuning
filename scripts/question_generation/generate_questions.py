import os
import json
import time
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================

CHUNKS_FILE = Path("data/chunks/selected_chunks_fixed.csv")
OUTPUT_FILE = Path("data/training/generated_pairs.csv")
LOG_FILE = Path("logs/question_generation.log")

# MODEL_NAME = "gemini-flash-latest"
MODEL_NAME = "gemini-3.1-flash-lite"


SAVE_EVERY = 1
MAX_RETRIES = 2
SLEEP_BETWEEN_REQUESTS = 10
BATCH_SIZE = 20

PROMPT_TEMPLATE = """
आप भारतीय कृषि विशेषज्ञ और RAG embedding dataset creator हैं।

नीचे दिए गए प्रत्येक कृषि passage से EXACTLY 3 प्राकृतिक हिंदी किसान-शैली के प्रश्न बनाइए।

उद्देश्य:
इन प्रश्नों का उपयोग MuRIL embedding model के retrieval fine-tuning के लिए होगा।
लक्ष्य यह है कि जब कोई किसान किसी अलग तरीके से वही जानकारी खोजे, तब भी मॉडल उसी passage को सही ढंग से retrieve कर सके।
इसलिए हर प्रश्न एक वास्तविक "search query" जैसा होना चाहिए।

Rules:

- हर प्रश्न का उत्तर केवल दिए गए passage में शब्दशः (verbatim) मौजूद होना चाहिए।
- Passage में जिस चीज़, रसायन, विधि, संख्या का जिक्र नहीं है, उसे प्रश्न में कभी मत डालिए — भले ही वह सामान्यतः उस विषय से जुड़ी लगती हो।
  उदाहरण: यदि passage में केवल "जाल और ढक्कन से ढकें" लिखा है और "पंचगव्य" शब्द कहीं नहीं है, तो प्रश्न में पंचगव्य का नाम मत लीजिए।
- प्रश्न छोटे, प्राकृतिक और स्पष्ट होने चाहिए।
- तीनों प्रश्न passage के अलग-अलग तथ्यों/हिस्सों को cover करें।

=== प्रश्न की विशिष्टता (SPECIFICITY) — महत्वपूर्ण ===

- हमेशा सबसे संकीर्ण (narrow) संभव प्रश्न पूछें, सबसे व्यापक (broad) नहीं।
- यदि passage में कोई एक योजना/तकनीक/उपाय का विशेष नाम है (जैसे "बीज ग्राम योजना", "मिनीकिट कार्यक्रम"), तो उसी विशेष नाम को लेकर प्रश्न बनाएं — "किसानों को कौन-कौन सी सहायता मिलती है" जैसे सामान्य प्रश्न मत बनाएं।

  ❌ गलत (बहुत व्यापक): "अरहर की उन्नत खेती के लिए किसानों को कौन-कौन सी सरकारी सहायताएँ मिलती हैं?"
  ✅ सही (संकीर्ण): "बीज ग्राम योजना के अंतर्गत किसानों को कितनी आर्थिक सहायता दी जाती है?"

- यदि passage में किसी काम के लिए 3+ चीज़ों/तरीकों/रसायनों की सूची है (जैसे "बीज को इन 5 चीजों से उपचारित करें"), तो:
  a) या तो पूरी सूची के बारे में एक ही प्रश्न पूछें ("बीज उपचार के लिए किन-किन चीजों का प्रयोग किया जाता है?") — यह स्वीकार्य है क्योंकि उत्तर एक इकाई (list) है,
  b) या सूची में से किसी एक विशिष्ट आइटम को नाम से लेकर प्रश्न बनाएं (यदि वह आइटम अकेले भी अर्थपूर्ण जानकारी देता है)।
  लेकिन एक साथ 2 अलग-अलग सूचियों/विषयों को मिलाकर एक प्रश्न मत बनाएं।

तीन प्रश्न इस प्रकार होने चाहिए:

1. Factual/decision question — विशिष्ट कृषि तथ्य पर आधारित (किस्म, मात्रा, विधि, रोग/कीट पहचान), narrow और specific।
2. Procedural question — "कैसे / कब / किस तरीके से" कोई विशेष कार्य किया जाए।
3. Semantic/alternate-phrasing/Natural farmer search question — वही मुख्य जानकारी अलग शब्दों में, फिर भी उतनी ही specific।

Topic/Crop Mention Rule:
- हर प्रश्न में मुख्य फसल/पशु/कीट/रोग/तकनीक का नाम अवश्य शामिल करें।

=== सख्ती से मना (STRICT REJECT LIST) ===

❌ किसी व्यक्ति/किसान का नाम पूछना, या किसने कितने में बेचा/कमाया
❌ किसी योजना/संस्था/जिले का नाम सिर्फ पहचान के लिए पूछना जब वह तकनीक न सिखाए
❌ पुरस्कार, सम्मान, वर्ष केवल record के तौर पर पूछना
❌ "इस लेख में क्या बताया गया है" जैसे सामान्य प्रश्न
❌ Passage की भाषा को लगभग वैसी की वैसी कॉपी करके प्रश्न बनाना
❌ ऐसा कोई शब्द/नाम/संख्या प्रश्न में डालना जो passage में शब्दशः मौजूद नहीं है (hallucination)
❌ बहुत व्यापक प्रश्न जो passage के आधे हिस्से का सारांश मांगे ("क्या-क्या सहायता मिलती है", "क्या-क्या फायदे हैं" — जब तक कि वह passage का एकमात्र/मुख्य बिंदु न हो)
❌ एक ही प्रश्न में दो अलग तथ्य/संख्याएं पूछना — "कितने दिन बाद X और Y में क्या मिलाएं" जैसे double-barrel questions; इन्हें दो अलग प्रश्नों में तोड़ें
❌ प्रश्न में अनावश्यक explanation/context जोड़ना — "जिसके शरीर पर घने बाल होने के कारण" जैसी phrases; प्रश्न सीधा और छोटा रखें

सोचने का तरीका (self-test, हर प्रश्न पर लागू करें):
1. "क्या कोई किसान यह प्रश्न असल ज़िंदगी में Google/YouTube पर टाइप करेगा?"
2. "क्या इस प्रश्न में मौजूद हर शब्द/नाम/संख्या passage में शब्दशः लिखा है?" — अगर नहीं, तो प्रश्न हटाकर दोबारा बनाएं।
3. "क्या यह प्रश्न इतना specific है कि इसका सिर्फ एक ही स्पष्ट उत्तर वाला हिस्सा passage में है?" — अगर उत्तर पूरे passage में फैला है, तो प्रश्न को संकीर्ण करें।

=== SKIP CONDITIONS ===

यदि पूरा passage निम्न में से किसी प्रकार का है, तो केवल "SKIP" लौटाएं:
- FAQ शैली का लेख
- किसी व्यक्ति/किसान की सफलता की कहानी, जिसमें कोई transferable तकनीक न हो
- सामान्य योजना announcement जिसमें कोई खेती तकनीक की जानकारी न हो

यदि passage आंशिक रूप से profile/story है लेकिन तकनीकी जानकारी भी मौजूद है — केवल तकनीकी हिस्से से प्रश्न बनाएं।

Quality Check (उत्तर देने से पहले हर प्रश्न पर verify करें):
1. क्या तीनों प्रश्न अलग-अलग जानकारी खोजते हैं?
2. क्या हर प्रश्न का हर शब्द/संख्या/नाम passage में शब्दशः मौजूद है? (कोई hallucination नहीं)
3a. क्या कोई प्रश्न बहुत व्यापक है? यदि हां, उसे संकीर्ण करें।
3b. क्या कोई प्रश्न में दो facts एक साथ पूछे गए हैं? यदि हां, उन्हें दो अलग प्रश्नों में तोड़ें और सबसे informative एक को रखें।
3c. क्या प्रश्न 15 शब्दों से अधिक लंबा है? यदि हां, अनावश्यक context/explanation हटाकर छोटा करें।
4. क्या कोई प्रश्न STRICT REJECT LIST में आता है?
5. क्या यह सच में एक किसान का specific search query जैसा लगता है?

Return ONLY valid JSON.

Format:
[
  {{
    "chunk_id":"...",
    "questions":[
      "...",
      "...",
      "..."
    ]
  }}
]

Passages:

{chunks}
"""
# ============================================================
# SETUP
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY is None:
    raise ValueError("GEMINI_API_KEY not found.")

client = genai.Client(api_key=API_KEY)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ============================================================
# HELPERS
# ============================================================

FAQ_KEYWORDS = [
    "अक्सर पूछे जाने वाले प्रश्न",
    "faq",
    "frequently asked questions",
]


def is_faq(row):

    fields = [
        str(row.get("title", "")),
        str(row.get("url", "")),
        str(row.get("domain", "")),
        str(row.get("article_id", "")),
    ]

    text = " ".join(fields).lower()

    keywords = [
        "अक्सर पूछे जाने वाले प्रश्न",
        "faq",
        "frequently asked questions",
    ]

    return any(k.lower() in text for k in keywords)

def load_processed():

    if not OUTPUT_FILE.exists():
        return set()

    df = pd.read_csv(OUTPUT_FILE)

    if len(df) == 0:
        return set()

    return set(df.chunk_id.astype(str))


def call_gemini(prompt):

    for attempt in range(MAX_RETRIES):

        try:
            print("Calling Gemini attempt:", attempt + 1)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.9,
                    max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0
                    ),
                    response_mime_type="application/json",
                ),
            )

            print("STOP REASON:", response.candidates[0].finish_reason)
            if not response.text:
                raise Exception("Gemini returned empty response")

            text = response.text.strip()
            print("=" * 60)
            print(text)
            print("=" * 60)         


            if text.upper() == "SKIP":
                return "SKIP"

            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

            decoder = json.JSONDecoder()
            result, end = decoder.raw_decode(text)

            if not isinstance(result, list):
                raise Exception("Output not list")

            return result

        except Exception as e:

            error = str(e).lower()

            if "429" in error:
                wait = 30
            elif "quota" in error:
                wait = 60
            elif "json" in error:
                wait = 5
            else:
                wait = 2 ** attempt

            print(f"ERROR attempt {attempt+1}: {e}")

            logging.warning(f"Retry {attempt+1} after {wait}s | {e}")
            time.sleep(wait)

            continue
    
    return None

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading chunks...")

df = pd.read_csv(CHUNKS_FILE)

processed = load_processed()

print(f"Total chunks      : {len(df)}")
print(f"Already processed : {len(processed)}")

rows = []

# ============================================================
# PROCESS
# ============================================================

for i in tqdm(range(0, len(df), BATCH_SIZE)):

    batch_df = df.iloc[i:i+BATCH_SIZE]

    batch = []
    batch_lookup = {}

    # -----------------------------------
    # Prepare batch
    # -----------------------------------

    for _, row in batch_df.iterrows():

        chunk_id = str(row["chunk_id"])

        if chunk_id in processed:
            continue

        if is_faq(row):
            logging.info(f"Skipped FAQ : {chunk_id}")
            continue

        chunk = str(row["chunk_text"])

        if not chunk.strip():
            logging.warning(f"Empty chunk : {chunk_id}")
            continue

        if len(chunk.split()) < 40:
            logging.info(f"Too short : {chunk_id}")
            continue

        batch_lookup[chunk_id] = row

        batch.append(
            f"""
Chunk ID: {chunk_id}

Title:
{row['title']}

Passage:
{chunk}
"""
        )

    if not batch:
        continue

    # -----------------------------------
    # Build prompt
    # -----------------------------------

    prompt = PROMPT_TEMPLATE.format(
        chunks="\n\n====================\n\n".join(batch)
    )

    # -----------------------------------
    # Generate questions
    # -----------------------------------

    result = call_gemini(prompt)

    if result is None:
        logging.error("Gemini request failed.")
        continue

    if result == "SKIP":
        continue

    # -----------------------------------
    # Parse response
    # -----------------------------------

    for item in result:

        chunk_id = str(item["chunk_id"])

        if chunk_id not in batch_lookup:
            logging.warning(f"Unknown chunk_id: {chunk_id}")
            continue

        row = batch_lookup[chunk_id]

        chunk = row["chunk_text"]
        title = row["title"]

        questions = list(
            dict.fromkeys(
                q.strip()
                for q in item["questions"]
                if q.strip()
            )
        )

        if len(questions) != 3:
            logging.warning(
                f"Wrong question count for {chunk_id}: {len(questions)}"
            )
            continue

        for q in questions:

            rows.append(
                {
                    "question": q,
                    "positive": chunk,
                    "chunk_id": chunk_id,
                    "document_id": row["document_id"],
                    "title": title,
                    "domain": row["domain"],
                    "language": row["language"],
                    "source": row["source"],
                    "url": row["url"],
                }
            )

        processed.add(chunk_id)

    # -----------------------------------
    # Save batch
    # -----------------------------------

    if rows:

        temp = pd.DataFrame(rows)

        if OUTPUT_FILE.exists():

            temp.to_csv(
                OUTPUT_FILE,
                mode="a",
                index=False,
                header=False,
                encoding="utf-8-sig",
            )

        else:

            temp.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig",
            )

        logging.info(f"Saved {len(rows)} rows")

        rows = []

    time.sleep(SLEEP_BETWEEN_REQUESTS)

# ============================================================
# SAVE REMAINING
# ============================================================

if rows:

    temp = pd.DataFrame(rows)

    if OUTPUT_FILE.exists():

        temp.to_csv(
            OUTPUT_FILE,
            mode="a",
            index=False,
            header=False,
            encoding="utf-8-sig",
        )

    else:

        temp.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

print("=" * 60)
print("Question generation completed.")
print(f"Saved to : {OUTPUT_FILE}")
logging.info("Completed successfully.") 

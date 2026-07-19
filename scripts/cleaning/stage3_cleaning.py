# import pandas as pd
# import re

# # ==========================================================
# # Paths
# # ==========================================================

# INPUT_FILE = "data/cleaned/keep_stage2_final.csv"
# OUTPUT_FILE = "data/cleaned/keep_stage3.csv"
# REMOVED_FILE = "data/cleaned/removed_stage3.csv"

# # ==========================================================
# # Load
# # ==========================================================

# df = pd.read_csv(INPUT_FILE)

# # ==========================================================
# # Rule Keywords
# # ==========================================================

# BAD_QUESTION_KEYWORDS = [

#     # Training / budget
#     "प्रशिक्षण शुल्क",
#     "बजट",
#     "अनुदान",
#     "सब्सिडी",
#     "राशि",
#     "कितने करोड़",
#     "प्रति किसान",
#     "प्रति समूह",
#     "प्रति प्रशिक्षण",
#     "प्रति दिन",
#     "आदमी दिवस",
#     "यात्रा भत्ता",
#     "खाना-पीना",
#     "आना-जाना",

#     # Administration
#     "समिति",
#     "मंत्रालय",
#     "प्रधानमंत्री",
#     "मसौदा",
#     "अपलोड",
#     "प्रोजेक्ट",
#     "योजना का गठन",

#     # Historical / document-specific
#     "किस वर्ष",
#     "कब लाया गया",
#     "कब जारी",
#     "बजट अनुमान",

#     # Very location specific
#     "गाँव",
#     "गांव",
#     "जिला",
#     "समस्तीपुर",
#     "हाजीपुर",
#     "शहजादापुर",

# ]

# LOCATION_WORDS = [

#     "बिहार",
#     "झारखंड",
#     "हरियाणा",
#     "पंजाब",
#     "गुजरात",
#     "समस्तीपुर",
#     "हाजीपुर",
#     "शहजादापुर",

# ]

# BAD_PASSAGE_KEYWORDS = [

#     "₹",
#     "रु.",
#     "प्रति किसान",
#     "प्रति समूह",
#     "यात्रा भत्ता",
#     "खाना-पीना",
#     "आना-जाना",
#     "बैच",
#     "आदमी दिवस",

# ]

# BAD_REASON_KEYWORDS = [

#     "administrative",
#     "budget",
#     "training subsidy",
#     "committee",
#     "government announcement",
#     "organization",
#     "location specific",
#     "historical",

# ]

# # ==========================================================
# # Helper
# # ==========================================================

# def contains(text, words):

#     text = str(text).lower()

#     for w in words:
#         if w.lower() in text:
#             return True

#     return False


# # ==========================================================
# # Boolean Filters
# # ==========================================================

# remove = (

#     (df["person_specific"] == True)
#     |
#     (df["identity_lookup"] == True)
#     |
#     (df["contact_information"] == True)
#     |
#     (df["navigation_or_metadata"] == True)
#     |
#     (df["encoding_problem"] == True)
#     |
#     (df["question_answer_match"] == False)

# )

# # ==========================================================
# # Agriculture & reusable
# # ==========================================================

# remove |= (df["agriculture"] == False)
# remove |= (df["reusable"] == False)

# # ==========================================================
# # Question keyword rules
# # ==========================================================

# remove |= df["question"].apply(
#     lambda x: contains(x, BAD_QUESTION_KEYWORDS)
# )

# # ==========================================================
# # Passage keyword rules
# # ==========================================================

# remove |= df["positive"].apply(
#     lambda x: contains(x, BAD_PASSAGE_KEYWORDS)
# )

# # ==========================================================
# # Location rules
# # ==========================================================

# remove |= df["question"].apply(
#     lambda x: contains(x, LOCATION_WORDS)
# )

# # ==========================================================
# # LLM reason
# # ==========================================================

# remove |= df["llm_reason"].fillna("").apply(
#     lambda x: contains(x, BAD_REASON_KEYWORDS)
# )

# # ==========================================================
# # Save
# # ==========================================================

# removed = df[remove]
# kept = df[~remove]

# removed.to_csv(
#     REMOVED_FILE,
#     index=False,
#     encoding="utf-8-sig"
# )

# kept.to_csv(
#     OUTPUT_FILE,
#     index=False,
#     encoding="utf-8-sig"
# )

# print("=" * 60)
# print("Original :", len(df))
# print("Removed  :", len(removed))
# print("Final    :", len(kept))
# print("=" * 60)









# Stage 3B

import re
import pandas as pd
from pathlib import Path

# ==========================================================
# FILES
# ==========================================================

INPUT_FILE = "data/cleaned/keep_stage3.csv"
OUTPUT_FILE = "data/cleaned/keep_stage3_final2.csv"
REMOVE_FILE = "data/cleaned/remove_stage3.csv"

# ==========================================================
# LOAD
# ==========================================================

df = pd.read_csv(INPUT_FILE)
removed_df = pd.DataFrame(columns=df.columns)

print("=" * 60)
print("Original :", len(df))
print("=" * 60)

# ==========================================================
# REMOVE USING LLM FLAGS
# ==========================================================

mask = (
    (df["reusable"] == True)
    & (df["question_answer_match"] == True)
    & (df["person_specific"] == False)
    & (df["identity_lookup"] == False)
    & (df["contact_information"] == False)
    & (df["navigation_or_metadata"] == False)
    & (df["encoding_problem"] == False)
)

removed_df = pd.concat([removed_df, df[~mask]], ignore_index=True)

df = df[mask].copy()

# ==========================================================
# REMOVE BY QUESTION PATTERNS
# ==========================================================

REMOVE_PATTERNS = [

    # ------------------------------------------------------
    # Government / Schemes
    # ------------------------------------------------------
    r"योजना",
    r"स्कीम",
    r"अनुदान",
    r"सब्सिडी",
    r"सहायता राशि",
    r"प्रोत्साहन राशि",
    r"बीमा",
    r"प्रीमियम",
    r"मानधन",
    r"ऋण",
    r"लोन",

    # ------------------------------------------------------
    # Financial
    # ------------------------------------------------------
    r"कितनी राशि",
    r"कितना अनुदान",
    r"कितनी सहायता",
    r"इकाई लागत",
    r"लागत",
    r"बजट",
    r"रुपये",
    r"₹",

    # ------------------------------------------------------
    # Administrative
    # ------------------------------------------------------
    r"प्रधानमंत्री",
    r"मुख्यमंत्री",
    r"मंत्रालय",
    r"विभाग",
    r"समिति",
    r"निगम",
    r"प्राधिकरण",
    r"एफपीओ",
    r"एफ\.एम\.सी\.एस",
    r"एनएमओओपी",
    r"आत्मा",
    r"बीज ग्राम",

    # ------------------------------------------------------
    # Application
    # ------------------------------------------------------
    r"आवेदन",
    r"कहाँ आवेदन",
    r"ऑफलाइन आवेदन",
    r"ऑनलाइन आवेदन",

    # ------------------------------------------------------
    # Training / Fees
    # ------------------------------------------------------
    r"प्रशिक्षण शुल्क",
    r"प्रति किसान",
    r"प्रति दिन",
    r"प्रति माह",
    r"प्रति वर्ष",
    r"मानदेय",
    r"वेतन",

    # ------------------------------------------------------
    # History
    # ------------------------------------------------------
    r"किस वर्ष",
    r"कब शुरू",
    r"स्थापना",
    r"गठन",
    r"उत्पत्ति",
    r"मुख्यालय",

    # ------------------------------------------------------
    # State / District specific
    # ------------------------------------------------------
    r"झारखंड",
    r"बिहार",
    r"हरियाणा",
    r"उत्तराखंड",
    r"गुजरात",
    r"राजस्थान",
    r"समस्तीपुर",
    r"हाजीपुर",
    r"ढूंडी",
    r"शहजादापुर",

    # ------------------------------------------------------
    # Administrative portals
    # ------------------------------------------------------
    r"पोर्टल",
    r"वेबसाइट",
    r"हेल्पलाइन",

]

pattern = re.compile("|".join(REMOVE_PATTERNS), flags=re.IGNORECASE)

mask = ~df["question"].fillna("").str.contains(pattern, regex=True)

removed_df = pd.concat([removed_df, df[~mask]], ignore_index=True)

df = df[mask]

# ==========================================================
# REMOVE DUPLICATE QUESTIONS
# ==========================================================

df["question"] = df["question"].str.strip()

dup_mask = df.duplicated(subset=["question"], keep="first")

removed_df = pd.concat([removed_df, df[dup_mask]], ignore_index=True)

df = df[~dup_mask]

# ==========================================================
# REMOVE VERY SHORT QUESTIONS
# ==========================================================

mask = df["question"].str.len() > 10

removed_df = pd.concat([removed_df, df[~mask]], ignore_index=True)

df = df[mask]

# ==========================================================
# REMOVE IF QUESTION ENDS WITH THESE
# ==========================================================

BAD_ENDINGS = (
    "किस वर्ष?",
    "कब शुरू हुआ?",
    "कब शुरू हुई?",
    "कहाँ स्थित है?",
    "कहाँ आवेदन करें?",
)

mask = ~df["question"].str.endswith(BAD_ENDINGS)

removed_df = pd.concat([removed_df, df[~mask]], ignore_index=True)

df = df[mask]

# ==========================================================
# REMOVE PASSAGES WITH MOSTLY NUMBERS
# ==========================================================

def too_many_numbers(text):

    text = str(text)

    numbers = len(re.findall(r"\d", text))

    chars = len(text)

    if chars == 0:
        return True

    return numbers / chars > 0.18


mask = ~df["positive"].apply(too_many_numbers)

removed_df = pd.concat([removed_df, df[~mask]], ignore_index=True)

df = df[mask]

# ==========================================================
# REMOVE PASSAGES WITH EXCESSIVE CURRENCY
# ==========================================================

currency_words = [
    "रुपये",
    "₹",
    "लाख",
    "करोड़",
    "प्रतिशत अनुदान",
]

mask = pd.Series(False, index=df.index)

for word in currency_words:
    mask |= df["positive"].str.contains(word, na=False)

removed_df = pd.concat([removed_df, df[mask]], ignore_index=True)

df = df[~mask]


print("=" * 60)
print("Removing duplicate Question-Passage pairs...")

before = len(df)

df = df.drop_duplicates(
    subset=["question", "positive"],
    keep="first"
)

after = len(df)

print(f"Before : {before}")
print(f"After  : {after}")
print(f"Removed: {before - after}")
print("=" * 60)


# ==========================================================
# SAVE
# ==========================================================

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

removed_df.to_csv(
    REMOVE_FILE,
    index=False,
    encoding="utf-8-sig",
)

print("=" * 60)
print("Original :", len(pd.read_csv(INPUT_FILE)))
print("Final    :", len(df))
print("Removed  :", len(pd.read_csv(INPUT_FILE)) - len(df))
print("=" * 60) 


print("Duplicate questions:", df.duplicated(subset=["question"]).sum())
dup_questions = df[df.duplicated(subset=["question"], keep=False)]

print("Number of duplicate rows:", len(dup_questions))

print(dup_questions[["question", "pair_id"]].sort_values("question")) 
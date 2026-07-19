import json
import time
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd
from google import genai

# ==========================================================
# CONFIG
# ==========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-lite"

INPUT_FILE = "data/cleaned/review_stage2.csv"

KEEP_FILE = "data/cleaned/keep_retry.csv"
REVIEW_FILE = "data/cleaned/review_retry.csv"
REMOVE_FILE = "data/cleaned/remove_retry.csv"

BATCH_SIZE = 50

SLEEP = 2
RETRY_SLEEP = 10      # seconds before retrying same batch
MAX_RETRIES = 5     # number of retries for same batch

client = genai.Client(api_key=API_KEY)

# ==========================================================
# PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are an expert dataset curator preparing a high-quality training dataset for fine-tuning MuRIL as a sentence embedding model for agricultural Retrieval-Augmented Generation (RAG).

This is NOT a chatbot evaluation task.

This is NOT a grammar task.

This is NOT a summarization task.

Your only objective is deciding whether each Question–Passage pair teaches reusable agricultural semantics that will improve dense retrieval.

------------------------------------------------------------
Dataset Goal
------------------------------------------------------------

The final embedding model should retrieve useful agricultural passages for future farmers.

Therefore KEEP only samples that teach reusable agricultural knowledge.

Reject samples that mainly memorize people, places, historical events, administrative facts or document-specific information.

------------------------------------------------------------
For EACH sample return EXACTLY ONE JSON object.

Return ONLY a JSON array.

[
{
"agriculture": true,
"reusable": true,
"person_specific": false,
"identity_lookup": false,
"location_specific": false,
"time_specific": false,
"contact_information": false,
"navigation_or_metadata": false,
"encoding_problem": false,
"question_answer_match": true,
"decision":"KEEP",
"reason":"short reason"
}
]

Do NOT output markdown.

Do NOT explain.

------------------------------------------------------------
Definitions

Agriculture

TRUE if the pair teaches knowledge related to

• crop cultivation
• horticulture
• livestock
• fisheries
• poultry
• soil
• irrigation
• fertilizer
• pest
• disease
• varieties
• harvesting
• storage
• processing
• machinery
• climate
• agricultural economics
• agricultural marketing
• government agricultural schemes
• farmer advisory
• food science
• post-harvest technology

FALSE otherwise.

------------------------------------------------------------

Reusable

TRUE only if a farmer from another state, district or year could reasonably ask the same question.

If the answer depends on a specific location, year, project, committee, event, or success story rather than general agricultural knowledge, set reusable = FALSE.

Examples:

✓ KEEP

How much nitrogen is present in urea?

Which soil is suitable for cashew cultivation?

How should stem borer be controlled?

When should fertilizer be applied?

Which variety is resistant to powdery mildew?

How should fish ponds be prepared?

------------------------------------------------------------

Reusable = FALSE if the answer mainly memorizes one document, one event or one historical fact.

Examples

Who received an award?

Which farmer did this?

Which village did this?

Which committee member?

Who inaugurated this?

Which district produced X in 2014?

How much was the ministry budget in 2018?

Where was the report uploaded?

Which organization prepared this report?

------------------------------------------------------------

Person_specific

TRUE if the answer mainly contains

farmer names

scientist names

officer names

committee members

beneficiaries

award winners

inventors

success stories

biographies

FALSE otherwise.

------------------------------------------------------------

Identity_lookup

TRUE if answering mainly requires identifying

person

organization

institution

committee

village

district

state

company

award

school

department

office

registration number

code

ID

or any list of named entities.

------------------------------------------------------------

Location_specific

TRUE if the answer is mainly about

one village

one district

one state

one farm

one institute

one project location

and does not teach generally reusable agricultural knowledge.

------------------------------------------------------------

Time_specific

TRUE if the answer mainly depends on

a specific year

budget

historical statistics

government target year

old production figures

survey year

one-time event

policy announcement

committee formation date

------------------------------------------------------------

Contact_information

TRUE if mainly contains

phone

email

address

website

office

helpline

contact person

------------------------------------------------------------

Navigation_or_metadata

TRUE if mainly contains

copyright

privacy

menu

table of contents

heading

footer

appendix

acknowledgement

image caption

document navigation

------------------------------------------------------------

Encoding_problem

TRUE if

OCR garbage

broken Unicode

random symbols

meaningless text

------------------------------------------------------------

Question_answer_match

TRUE only if the passage actually answers the question.

FALSE if unrelated or incomplete.

------------------------------------------------------------
Decision Rules

KEEP only if ALL are true

agriculture = true

reusable = true

question_answer_match = true

person_specific = false

identity_lookup = false

location_specific = false

time_specific = false

contact_information = false

navigation_or_metadata = false

encoding_problem = false

------------------------------------------------------------

REMOVE if ANY of these are true

person_specific

identity_lookup

location_specific

time_specific

contact_information

navigation_or_metadata

encoding_problem

question_answer_match = false

------------------------------------------------------------

Borderline examples

KEEP

What temperature is suitable for grape cultivation?

Which fertilizer should be applied?

How should stem borer be controlled?

How much nitrogen does urea contain?

Which soil is suitable for cashew?

How should fish ponds be managed?

------------------------------------------------------------

REMOVE

Which farmer received the award?

Which village sent people for training?

What was Bihar fish production in 2014?

Who was committee chairman?

What was the ministry budget in 2018?

Where was the draft report uploaded?

Which district won the award?

Who started this success story?

Which state had 42 lakh hectares in 2013?

REMOVE if the question is primarily about:

- a specific village
- a specific district
- a specific state project
- a specific institute's local project
- one-time government event
- one-time government announcement
- survey statistics for a particular year
- local success story
- demonstration farm
- pilot project
- training conducted at a specific location

unless the knowledge teaches a generally reusable agricultural practice.

------------------------------------------------------------

When uncertain between KEEP and REMOVE, choose REVIEW only if the sample still contains useful agricultural semantics but requires human inspection.

Otherwise choose REMOVE.

Your goal is maximizing retrieval quality, not maximizing dataset size.
"""

# ==========================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
    keep_default_na=False
)

# Retry rows that failed previously.
# Older failed rows have "api_error" shifted into the agriculture column.
mask = (
    df["llm_reason"].astype(str).str.startswith("api_error", na=False)
    | df["agriculture"].astype(str).str.startswith("api_error", na=False)
)

df = df[mask].reset_index(drop=True)

print(f"Retrying API error rows: {len(df)}")

# ==========================================================

for start in range(0, len(df), BATCH_SIZE):

    batch = df.iloc[start:start+BATCH_SIZE]

    prompt = ""

    for i, (_, row) in enumerate(batch.iterrows()):

        prompt += f"""

Sample {i}

Question:
{row['question']}

Passage:
{row['positive']}
"""

    retry = 0

    while True:

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=SYSTEM_PROMPT + prompt,
                config={
                    "temperature":0,
                    "top_p":0,
                    "response_mime_type":"application/json",
                }
            )

            print(response.text[:200])

            try:
                labels = json.loads(response.text)

                if len(labels) == len(batch) + 1:
                    print("Received one extra label.")
                    print("Extra label:")
                    print(json.dumps(labels[-1], indent=2, ensure_ascii=False))

                    labels = labels[:len(batch)]
                    assert len(labels) == len(batch)

                elif len(labels) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} labels, got {len(labels)}"
                    )

            except Exception as e:
                print(f"Response parsing error: {e}")
                raise

            print("Total pairs:", len(df))

            for (_, row), result in zip(batch.iterrows(), labels):

                Path("data/cleaned").mkdir(exist_ok=True)

                row = row.to_dict()

                # Save all LLM decisions
                row["agriculture"] = result.get("agriculture", False)
                row["reusable"] = result.get("reusable", False)
                row["person_specific"] = result.get("person_specific", False)
                row["identity_lookup"] = result.get("identity_lookup", False)
                row["contact_information"] = result.get("contact_information", False)
                row["navigation_or_metadata"] = result.get("navigation_or_metadata", False)
                row["encoding_problem"] = result.get("encoding_problem", False)
                row["question_answer_match"] = result.get("question_answer_match", False)
                row["llm_reason"] = result.get("reason", "")

                # ----------------------------
                # Decide label ourselves
                # ----------------------------

                if (
                    row["person_specific"]
                    or row["identity_lookup"]
                    or not row["question_answer_match"]
                    or row["contact_information"]
                    or row["navigation_or_metadata"]
                    or row["encoding_problem"]
                ):
                    pd.DataFrame([row]).to_csv(
                        REMOVE_FILE,
                        mode="a",
                        header=not Path(REMOVE_FILE).exists(),
                        index=False,
                        encoding="utf-8-sig",
                    )

                elif (
                    row["agriculture"]
                    and row["reusable"]
                    and row["question_answer_match"]
                ):
                    pd.DataFrame([row]).to_csv(
                        KEEP_FILE,
                        mode="a",
                        header=not Path(KEEP_FILE).exists(),
                        index=False,
                        encoding="utf-8-sig",
                    )

                else:
                    pd.DataFrame([row]).to_csv(
                        REVIEW_FILE,
                        mode="a",
                        header=not Path(REVIEW_FILE).exists(),
                        index=False,
                        encoding="utf-8-sig",
                    )

            break

        except Exception as e:

            retry += 1

            print(e)
            print(f"Retry {retry}/{MAX_RETRIES}")

            if retry >= MAX_RETRIES:

                print("Maximum retries reached. Moving batch to REVIEW.")

                for _, row in batch.iterrows():

                    row = row.to_dict()

                    # Fix corrupted rows where api_error ended up in agriculture
                    if row.get("agriculture") == "api_error":
                        row["agriculture"] = ""
                        row["reusable"] = ""
                        row["person_specific"] = ""
                        row["identity_lookup"] = ""
                        row["contact_information"] = ""
                        row["navigation_or_metadata"] = ""
                        row["encoding_problem"] = ""
                        row["question_answer_match"] = ""

                    row["llm_reason"] = f"api_error: {str(e)}"

                    pd.DataFrame([row]).to_csv(
                        REVIEW_FILE,
                        mode="a",
                        header=not Path(REVIEW_FILE).exists(),
                        index=False,
                        encoding="utf-8-sig",
                    )

                break

            print(f"Retrying same batch in {RETRY_SLEEP} seconds...")
            time.sleep(RETRY_SLEEP)
            
    print(f"Processed {min(start+BATCH_SIZE, len(df))}/{len(df)}")

    time.sleep(SLEEP)

# ==========================================================

Path("data/cleaned").mkdir(exist_ok=True)

print("=" * 60)
print("Cleaning completed.")
print("=" * 60)
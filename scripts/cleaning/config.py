from pathlib import Path

# ==========================================================
# API KEYS (rotate automatically)
# ==========================================================

API_KEYS = [
    "YOUR_API_KEY_1",
    "YOUR_API_KEY_2",
    "YOUR_API_KEY_3",
    "YOUR_API_KEY_4",
]

MODEL = "gemini-2.5-flash-lite"

# ==========================================================
# PATHS
# ==========================================================

INPUT_FILE = Path("data/cleaned/keep_stage1.csv")

BATCH_DIR = Path("data/llm_batches")
RESULT_DIR = Path("data/llm_results")

KEEP_FILE = Path("data/cleaned/keep_stage2.csv")
REVIEW_FILE = Path("data/cleaned/review_stage2.csv")
REMOVE_FILE = Path("data/cleaned/remove_stage2.csv")

LOG_FILE = Path("logs/stage2.log")

BATCH_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ==========================================================
# SETTINGS
# ==========================================================

BATCH_SIZE = 40

TEMPERATURE = 0.0

MAX_OUTPUT_TOKENS = 4096

MAX_RETRIES = 5

REQUEST_DELAY = 2

# ==========================================================
# PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are cleaning a dataset for contrastive retrieval training
(MNRL / Sentence Transformer).

Each sample contains:

- question
- positive passage

Your job is NOT to improve the text.

Only classify each sample.

Return one label:

KEEP
REVIEW
REMOVE

--------------------------------------------------------

KEEP

Good agricultural factual questions.

Examples

• Crop cultivation
• Pest management
• Fertilizer
• Irrigation
• Government schemes
• Animal husbandry
• Fisheries
• Soil
• Horticulture
• Farm machinery
• Storage
• Weather
• Organic farming
• Seed production
• Diseases
• Scientific agriculture

--------------------------------------------------------

REVIEW

Potentially useful but uncertain.

Examples

• Long wording
• Multiple questions
• Slight OCR noise
• Formatting issues
• Needs manual verification

--------------------------------------------------------

REMOVE

Questions that should NEVER be used.

Examples

Person names

Phone numbers

Addresses

Email IDs

Website navigation

Copyright

Feedback

Login

Gallery

Photo

Video

Breadcrumbs

Advertisement

Contact us

Office timing

Tender

Recruitment

Vacancy

Repeated boilerplate

Random metadata

Broken OCR

Pure greetings

Meaningless text

Navigation labels

Only years

Only numbers

--------------------------------------------------------

IMPORTANT

Do NOT remove agricultural factual questions
even if wording is imperfect.

Return ONLY valid JSON.

Format

[
 {
   "id":"...",
   "label":"KEEP"
 }
]

No markdown.
No explanation.
"""
from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "categories" / "pdf_links.txt"
OUTPUT_DIR = BASE_DIR / "data" / "filtered_urls"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USEFUL_FILE = OUTPUT_DIR / "useful_pdf.txt"
REVIEW_FILE = OUTPUT_DIR / "review_pdf.txt"
SKIPPED_FILE = OUTPUT_DIR / "skipped_pdf.txt"

# ==========================================================
# Keywords
# ==========================================================

KEEP_KEYWORDS = [
    "kheti",
    "indian farming",
    "indian horticulture",

    "agriculture",
    "agri",

    "crop",
    "seed",
    "soil",
    "fertilizer",
    "fertiliser",
    "irrigation",

    "pest",
    "disease",
    "weed",

    "horticulture",
    "fruit",
    "vegetable",

    "technology",
    "guide",
    "manual",
    "handbook",
    "package",

    "research",
    "traditional",
    "indigenous",
    "innovation",
    "itk",

    "farmer"
]

SKIP_KEYWORDS = [
    "annual report",
    "annual reports",

    "achievement",
    "achievements",

    "monthly summary",
    "monthly achievements",

    "advertisement",
    "vacancy",
    "recruitment",
    "career",

    "circular",
    "notice",

    "audit",
    "accounts",

    "telephone",
    "directory",

    "rti",

    "pension",

    "office order",

    "tender",

    "press release",

    "newsletter",

    "photo",
    "gallery",

    "video",

    "event",

    "minutes",

    "meeting",

    "procurement",

    "finance"
]

# ==========================================================
# Helpers
# ==========================================================

def save(file_path, url):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(url + "\n")

# ==========================================================
# Main
# ==========================================================

def main():

    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]

    print(f"Loaded {len(urls)} PDF URLs\n")

    for url in urls:

        u = url.lower()

        # -----------------------------
        # 1. SKIP (highest priority)
        # -----------------------------
        if any(k in u for k in SKIP_KEYWORDS):
            save(SKIPPED_FILE, url)
            print("SKIP  ->", url)
            continue

        # -----------------------------
        # 2. KEEP
        # -----------------------------
        elif any(k in u for k in KEEP_KEYWORDS):
            save(USEFUL_FILE, url)
            print("KEEP  ->", url)

        # -----------------------------
        # 3. REVIEW
        # -----------------------------
        else:
            save(REVIEW_FILE, url)
            print("REVIEW ->", url)

    print("\nDone!")
    print(f"Useful : {USEFUL_FILE}")
    print(f"Review : {REVIEW_FILE}")
    print(f"Skipped: {SKIPPED_FILE}")


if __name__ == "__main__":
    main()
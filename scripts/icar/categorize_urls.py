#!/usr/bin/env python3
"""
categorize_urls.py

Reads URLs from:
    data/urls/icar_all_urls.txt

Visits each page and extracts:
    - Page title
    - Meta description
    - PDF links

Categorizes URLs into:
    - reports.txt
    - books.txt
    - magazines.txt
    - publications.txt
    - html_articles.txt
    - skipped.txt
    - failed_urls.txt
    - pdf_links.txt
"""

from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = BASE_DIR / "data" / "urls" / "icar_all_urls.txt"
OUTPUT_DIR = BASE_DIR / "data" / "categories"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# HTTP Settings
# ==========================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    )
}

TIMEOUT = 20

session = requests.Session()
session.headers.update(HEADERS)

# ==========================================================
# Category Rules
# ==========================================================

CATEGORY_RULES = {
    "reports": [
        "annual report",
        "report",
        "dare",
    ],
    "magazines": [
        "kheti",
        "indian farming",
        "indian horticulture",
        "magazine",
        "journal",
    ],
    "books": [
        "book",
        "ebook",
        "manual",
        "guide",
    ],
    "publications": [
        "publication",
        "catalogue",
        "bulletin",
    ],
}

# ==========================================================
# Skip Keywords
# ==========================================================

SKIP_KEYWORDS = [
    "contact",
    "vacancy",
    "career",
    "tender",
    "telephone",
    "feedback",
    "privacy",
    "copyright",
    "login",
    "webmail",
    "account",
    "audit",
    "notification",
    "circular",
    "notice",
    "gallery",
    "video",
    "photo",
    "event",
]

# ==========================================================
# Utility Functions
# ==========================================================


def save_to_file(filename: str, text: str):
    """Append one line to an output file."""

    outfile = OUTPUT_DIR / filename

    with open(outfile, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def extract_page_info(url):
    """
    Returns:
        title
        meta_description
        pdf_links
    """

    try:
        response = session.get(
            url,
            timeout=TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

    except Exception:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # ------------------------------------------------------
    # Title
    # ------------------------------------------------------

    title = ""

    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # ------------------------------------------------------
    # Meta Description
    # ------------------------------------------------------

    meta_description = ""

    meta = soup.find("meta", attrs={"name": "description"})

    if not meta:
        meta = soup.find("meta", attrs={"property": "og:description"})

    if meta:
        meta_description = meta.get("content", "").strip()

    # ------------------------------------------------------
    # PDF Links
    # ------------------------------------------------------

    pdf_links = set()

    for a in soup.find_all("a", href=True):

        href = a["href"]

        full_url = urljoin(url, href)

        if full_url.lower().endswith(".pdf"):
            pdf_links.add(full_url)

    return title, meta_description, sorted(pdf_links)


# ==========================================================
# Main
# ==========================================================


def main():

    if not INPUT_FILE.exists():
        print(f"Input file not found:\n{INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]

    urls = sorted(set(urls))

    print("=" * 60)
    print(f"Loaded {len(urls)} URLs")
    print("=" * 60)

    for index, url in enumerate(urls, start=1):

        print(f"[{index}/{len(urls)}] {url}")

        # --------------------------------------------------
        # Skip obvious junk URLs
        # --------------------------------------------------

        url_lower = url.lower()

        if any(word in url_lower for word in SKIP_KEYWORDS):
            save_to_file("skipped.txt", url)
            print("   Skipped")
            continue

        # --------------------------------------------------
        # Download page
        # --------------------------------------------------

        result = extract_page_info(url)

        if result is None:
            save_to_file("failed_urls.txt", url)
            print("   Failed")
            continue

        title, meta_description, pdf_links = result

        # --------------------------------------------------
        # Save PDF Links
        # --------------------------------------------------

        for pdf in pdf_links:
            save_to_file("pdf_links.txt", pdf)

        # --------------------------------------------------
        # Searchable Text
        # --------------------------------------------------

        page_text = " ".join([
            title,
            meta_description,
            url
        ]).lower()

        # --------------------------------------------------
        # Categorize
        # --------------------------------------------------

        matched = []

        for category, keywords in CATEGORY_RULES.items():

            if any(keyword in page_text for keyword in keywords):
                matched.append(category)

        # --------------------------------------------------
        # Save Categories
        # --------------------------------------------------

        if matched:

            for category in matched:
                save_to_file(f"{category}.txt", url)

            print("   → " + ", ".join(matched))

        else:

            save_to_file("html_articles.txt", url)
            print("   → html_articles")

    print("\n" + "=" * 60)
    print("Finished Successfully")
    print(f"Output Folder : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
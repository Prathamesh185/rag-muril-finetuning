from pathlib import Path
import xml.etree.ElementTree as ET

SITEMAP_FILE = Path("data/sitemap/sitemap.xml")
OUTPUT_FILE = Path("data/urls/icar_all_urls.txt")

root = ET.parse(SITEMAP_FILE).getroot()

namespace = {
    "ns": "http://www.sitemaps.org/schemas/sitemap/0.9"
}

urls = []

for url in root.findall("ns:url", namespace):
    loc = url.find("ns:loc", namespace)

    if loc is not None:
        urls.append(loc.text.strip())

urls = sorted(set(urls))

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for url in urls:
        f.write(url + "\n")

print(f"Extracted {len(urls)} URLs.")
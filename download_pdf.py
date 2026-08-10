import requests

# Target the direct upload folder URL, not the viewer.html endpoint
pdf_url = "https://agrigyan.in/hindi/wp-content/uploads/chapter-1-agronomy-soil-and-seed-agrigyan.pdf"
output_filename = "chapter_1_agronomy.pdf"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

print("Downloading protected file source...")
response = requests.get(pdf_url, headers=headers)

if response.status_code == 200:
    with open(output_filename, "wb") as f:
        f.write(response.content)
    print(f"Success! Saved clean, selectable PDF as: {output_filename}")
else:
    print(f"Download failed. Status Code: {response.status_code}")
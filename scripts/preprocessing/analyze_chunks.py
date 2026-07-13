import pandas as pd

# Load chunks
df = pd.read_csv("data/chunks/vikaspedia_passages.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Total chunks : {len(df)}")

# Count chunks per article
article_stats = (
    df.groupby("document_id")
      .size()
      .reset_index(name="num_chunks")
)

print(f"Total articles : {len(article_stats)}")

print("\nChunk distribution\n")

print(article_stats["num_chunks"].describe())

print("\nArticles by chunk count")

ranges = {
    "1 chunk": (1, 1),
    "2-3 chunks": (2, 3),
    "4-6 chunks": (4, 6),
    "7-10 chunks": (7, 10),
    "11-20 chunks": (11, 20),
    "21+ chunks": (21, 1000)
}

for name, (low, high) in ranges.items():
    count = article_stats[
        (article_stats["num_chunks"] >= low) &
        (article_stats["num_chunks"] <= high)
    ].shape[0]

    print(f"{name:<12}: {count}")

print("\nLargest articles\n")

print(
    article_stats
    .sort_values("num_chunks", ascending=False)
    .head(20)
)
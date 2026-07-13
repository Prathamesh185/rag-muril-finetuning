import pandas as pd

df = pd.read_csv("data/chunks/selected_chunks_fixed.csv")

for _, row in df[df.word_count > 400].iterrows():
    print("=" * 80)
    print(row["chunk_id"])
    print(row["word_count"])
    print(row["chunk_text"][:1000])
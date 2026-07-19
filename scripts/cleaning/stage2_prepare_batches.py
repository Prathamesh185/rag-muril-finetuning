import pandas as pd
from pathlib import Path

from config import INPUT_FILE, BATCH_DIR, BATCH_SIZE


def main():

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} pairs")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    total_batches = 0

    for i in range(0, len(df), BATCH_SIZE):

        batch = df.iloc[i:i + BATCH_SIZE]

        outfile = BATCH_DIR / f"batch_{total_batches:04d}.csv"

        batch.to_csv(outfile, index=False, encoding="utf-8-sig")

        total_batches += 1

    print("=" * 60)
    print(f"Total pairs   : {len(df)}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Total batches : {total_batches}")
    print(f"Saved to      : {BATCH_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
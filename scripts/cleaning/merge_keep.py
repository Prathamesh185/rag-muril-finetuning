import pandas as pd

# Read both files
keep_stage2 = pd.read_csv(
    "data/cleaned/keep_stage2.csv",
    encoding="utf-8-sig"
)

keep_retry = pd.read_csv(
    "data/cleaned/keep_retry.csv",
    encoding="utf-8-sig"
)

print("Original keep:", len(keep_stage2))
print("Retry keep:", len(keep_retry))

# Merge
merged = pd.concat([keep_stage2, keep_retry], ignore_index=True)

# Remove duplicates based on pair_id
merged = merged.drop_duplicates(subset="pair_id")

print("Final keep:", len(merged))

# Save as a new file
merged.to_csv(
    "data/cleaned/keep_stage2_final.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Merged file saved as data/cleaned/keep_stage2_final.csv")
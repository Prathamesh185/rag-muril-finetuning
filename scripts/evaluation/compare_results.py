import json
from pathlib import Path

import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "output"

BASE_RESULTS = OUTPUT_DIR / "base_results.json"
FINETUNED_RESULTS = OUTPUT_DIR / "finetuned_results.json"


# ==========================================================
# LOAD RESULTS
# ==========================================================

with open(BASE_RESULTS, "r", encoding="utf-8") as file:
    base_results = json.load(file)

with open(FINETUNED_RESULTS, "r", encoding="utf-8") as file:
    finetuned_results = json.load(file)


# ==========================================================
# METRICS TO COMPARE
# ==========================================================

metrics = [
    "accuracy@1",
    "accuracy@3",
    "accuracy@5",
    "accuracy@10",
    "precision@1",
    "precision@3",
    "precision@5",
    "precision@10",
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "mrr@10",
    "ndcg@10",
    "map@10",
]


# ==========================================================
# FIND METRIC
# ==========================================================

def find_metric(results, metric_name):
    for key, value in results.items():
        key = key.lower()

        if "cosine" in key and key.endswith(metric_name):
            return float(value)

    return None


# ==========================================================
# BUILD COMPARISON
# ==========================================================

rows = []

for metric in metrics:
    base_value = find_metric(
        base_results,
        metric,
    )

    finetuned_value = find_metric(
        finetuned_results,
        metric,
    )

    if base_value is None or finetuned_value is None:
        print(f"Warning: {metric} not found")
        continue

    absolute_gain = finetuned_value - base_value

    relative_gain = (
        (absolute_gain / base_value) * 100
        if base_value != 0
        else None
    )

    rows.append(
        {
            "Metric": metric.upper(),
            "Base MuRIL": base_value,
            "Fine-tuned MuRIL": finetuned_value,
            "Absolute Gain": absolute_gain,
            "Relative Gain (%)": relative_gain,
        }
    )


comparison = pd.DataFrame(rows)


# ==========================================================
# SAVE COMPARISON
# ==========================================================

comparison_file = OUTPUT_DIR / "model_comparison.csv"

comparison.to_csv(
    comparison_file,
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f",
)


# ==========================================================
# DISPLAY COMPARISON
# ==========================================================

display = comparison.copy()

for column in [
    "Base MuRIL",
    "Fine-tuned MuRIL",
    "Absolute Gain",
]:
    display[column] = display[column].map(
        lambda value: f"{value:.4f}"
    )

display["Relative Gain (%)"] = display[
    "Relative Gain (%)"
].map(
    lambda value: (
        f"{value:.2f}%"
        if pd.notna(value)
        else "N/A"
    )
)


print("\n" + "=" * 95)
print("BASE MuRIL vs FINE-TUNED MuRIL")
print("=" * 95)

print(
    display.to_string(
        index=False,
    )
)

print("=" * 95)

print("\nSaved to:")
print(comparison_file)
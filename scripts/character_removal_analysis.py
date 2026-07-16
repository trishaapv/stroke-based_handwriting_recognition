import numpy as np
import pandas as pd

from shapley import (
    DEVICE,
    coalition_sample,
    load_point_model,
    load_point_test_split,
    payoff,
)

samples, classes, label_to_index, labels, test_idx = load_point_test_split()
model = load_point_model(num_classes=len(classes), device=DEVICE)
shapley_df = pd.read_csv("shapley_results_clean.csv")

results = []

for original_idx in test_idx:
    sample = samples[original_idx]
    target = int(labels[original_idx])
    char = classes[target]

    if len(sample["strokes"]) < 2:
        continue

    row = shapley_df[shapley_df["sample_id"] == original_idx]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    shapley_vals = np.array([row["stroke_1"], row["stroke_2"], row["stroke_3"]])
    real_strokes = list(range(len(sample["strokes"])))
    valid_vals = shapley_vals[real_strokes]

    high_idx = real_strokes[np.argmax(valid_vals)]
    low_idx = real_strokes[np.argmin(valid_vals)]

    full_conf = payoff(model, sample, target, DEVICE)
    high_removed = coalition_sample(sample, set(real_strokes) - {high_idx})
    low_removed = coalition_sample(sample, set(real_strokes) - {low_idx})

    high_drop = full_conf - payoff(model, high_removed, target, DEVICE)
    low_drop = full_conf - payoff(model, low_removed, target, DEVICE)

    results.append({
        "character": char,
        "high_drop": high_drop,
        "low_drop": low_drop,
    })

df = pd.DataFrame(results)
summary = df.groupby("character").mean().reset_index()
summary["ratio"] = summary["high_drop"] / (summary["low_drop"] + 1e-8)
summary = summary.sort_values("ratio", ascending=False)

print("\nHIGHEST RATIOS")
print("=" * 50)
print(summary.head(15))

print("\nLOWEST RATIOS")
print("=" * 50)
print(summary.tail(15))

summary.to_csv("character_removal_summary.csv", index=False)
print("\nSaved character_removal_summary.csv")
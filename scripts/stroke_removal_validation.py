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
    real_strokes = list(range(len(sample["strokes"])))

    if len(real_strokes) <= 1:
        continue

    row = shapley_df[shapley_df["sample_id"] == original_idx]
    if len(row) == 0:
        continue
    row = row.iloc[0]

    shapley = np.array([row["stroke_1"], row["stroke_2"], row["stroke_3"]])
    real_vals = shapley[real_strokes]
    highest = real_strokes[np.argmax(real_vals)]
    lowest = real_strokes[np.argmin(real_vals)]

    original_conf = payoff(model, sample, target, DEVICE)
    conf_high = payoff(model, coalition_sample(sample, set(real_strokes) - {highest}), target, DEVICE)
    conf_low = payoff(model, coalition_sample(sample, set(real_strokes) - {lowest}), target, DEVICE)

    results.append({
        "sample_id": int(original_idx),
        "original_conf": original_conf,
        "drop_high": original_conf - conf_high,
        "drop_low": original_conf - conf_low,
    })

results = pd.DataFrame(results)
results.to_csv("stroke_removal_results.csv", index=False)

print("=" * 60)
print("Average Drop (Highest Shapley):", results["drop_high"].mean())
print("Average Drop (Lowest Shapley):", results["drop_low"].mean())
print("=" * 60)
print("\nSaved stroke_removal_results.csv")
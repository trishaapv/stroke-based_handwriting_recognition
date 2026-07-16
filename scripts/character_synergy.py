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

results = []

for original_idx in test_idx:
    sample = samples[original_idx]
    target = int(labels[original_idx])

    full_value = payoff(model, sample, target, DEVICE)
    single_sum = 0.0

    for stroke_idx in range(len(sample["strokes"])):
        single_sum += payoff(
            model,
            coalition_sample(sample, {stroke_idx}),
            target,
            DEVICE,
        )

    synergy = full_value - single_sum

    results.append({
        "sample_id": int(original_idx),
        "character": classes[target],
        "full_value": full_value,
        "single_sum": single_sum,
        "synergy": synergy,
    })

results = pd.DataFrame(results)

summary = (
    results
    .groupby("character")["synergy"]
    .mean()
    .reset_index()
    .sort_values("synergy", ascending=False)
)

print("\nMOST COOPERATIVE")
print("=" * 50)
print(summary.head(15))

print("\nLEAST COOPERATIVE")
print("=" * 50)
print(summary.tail(15))

summary.to_csv("character_synergy.csv", index=False)
results.to_csv("sample_synergy.csv", index=False)

print("\nSaved character_synergy.csv")
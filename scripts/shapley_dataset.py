import pandas as pd

from shapley import (
    DEVICE,
    load_point_model,
    load_point_test_split,
    predict,
    shapley_values,
    stroke_count,
)

samples, classes, label_to_index, labels, test_idx = load_point_test_split()
model = load_point_model(num_classes=len(classes), device=DEVICE)

rows = []

for original_idx in test_idx:
    sample = samples[original_idx]
    true_class = int(labels[original_idx])
    phi = shapley_values(model, sample, true_class, DEVICE)
    pred_class, confidence = predict(model, sample, DEVICE)

    rows.append({
        "sample_id": int(original_idx),
        "true_label": classes[true_class],
        "pred_label": classes[pred_class],
        "confidence": confidence,
        "num_strokes": stroke_count(sample),
        "stroke_1": phi[0],
        "stroke_2": phi[1],
        "stroke_3": phi[2],
    })

df = pd.DataFrame(rows)
df.to_csv("shapley_results_clean.csv", index=False)

print(df.head())
print("\nSaved shapley_results_clean.csv")
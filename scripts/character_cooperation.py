import pandas as pd
import numpy as np

df = pd.read_csv(
    "shapley_results_clean.csv"
)

# -------------------------
# Cooperation Index
# -------------------------

def cooperation_index(row):

    vals = np.array([
        row["stroke_1"],
        row["stroke_2"],
        row["stroke_3"]
    ])

    vals = vals[vals > 0]

    if len(vals) == 0:
        return 0

    return 1 - (
        np.max(vals)
        /
        np.sum(vals)
    )

df["cooperation_index"] = (
    df.apply(
        cooperation_index,
        axis=1
    )
)

# -------------------------
# Character averages
# -------------------------

char_summary = (
    df.groupby("true_label")
      ["cooperation_index"]
      .mean()
      .reset_index()
)

char_summary = (
    char_summary.sort_values(
        "cooperation_index",
        ascending=False
    )
)

print("\nMost Cooperative Characters")
print("=" * 40)

print(
    char_summary.head(10)
)

print("\nLeast Cooperative Characters")
print("=" * 40)

print(
    char_summary.tail(10)
)

char_summary.to_csv(
    "character_cooperation.csv",
    index=False
)

print(
    "\nSaved character_cooperation.csv"
)
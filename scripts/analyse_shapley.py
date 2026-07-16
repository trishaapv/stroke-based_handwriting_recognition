import pandas as pd
import numpy as np

df = pd.read_csv(
    "shapley_results_clean.csv"
)

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Samples:", len(df))

print("\nAverage Shapley Values")

print(
    df[
        ["stroke_1",
         "stroke_2",
         "stroke_3"]
    ].mean()
)

print("\n")


# =====================================
# DOMINANT STROKE
# =====================================

dominant = []

for _, row in df.iterrows():

    vals = np.array([
        row["stroke_1"],
        row["stroke_2"],
        row["stroke_3"]
    ])

    dominant.append(
        np.argmax(vals) + 1
    )

df["dominant_stroke"] = dominant

print("=" * 60)
print("DOMINANT STROKE COUNTS")
print("=" * 60)

print(
    df["dominant_stroke"]
    .value_counts()
)

print()


# =====================================
# CHARACTER-WISE ANALYSIS
# =====================================

print("=" * 60)
print("CHARACTER ANALYSIS")
print("=" * 60)

results = []

for char in sorted(
    df["true_label"].unique()
):

    subset = df[
        df["true_label"] == char
    ]

    mean_s1 = subset[
        "stroke_1"
    ].mean()

    mean_s2 = subset[
        "stroke_2"
    ].mean()

    mean_s3 = subset[
        "stroke_3"
    ].mean()

    results.append([
        char,
        mean_s1,
        mean_s2,
        mean_s3
    ])

char_df = pd.DataFrame(
    results,
    columns=[
        "character",
        "stroke_1",
        "stroke_2",
        "stroke_3"
    ]
)

char_df.to_csv(
    "character_shapley_summary.csv",
    index=False
)

print(
    char_df.head()
)

print(
    "\nSaved character_shapley_summary.csv"
)
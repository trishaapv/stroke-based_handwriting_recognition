import pandas as pd

df = pd.read_csv(
    "shapley_results_clean.csv"
)

targets = [
    "x",
    "N",
    "I",
    "l",
    "z",
    "b",
    "d",
    "Y"
]

for ch in targets:

    print("\n")
    print("="*50)
    print("Character:", ch)

    subset = df[
        df["true_label"] == ch
    ]

    print(
        subset[
            [
                "stroke_1",
                "stroke_2",
                "stroke_3"
            ]
        ].head()
    )
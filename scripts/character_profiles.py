import pandas as pd

# -------------------------
# LOAD FILES
# -------------------------

shapley = pd.read_csv(
    "shapley_results_clean.csv"
)

coop = pd.read_csv(
    "character_cooperation.csv"
)

synergy = pd.read_csv(
    "character_synergy.csv"
)

# -------------------------
# MEAN SHAPLEY PER CHARACTER
# -------------------------

stroke_summary = (
    shapley
    .groupby("true_label")
    [
        [
            "stroke_1",
            "stroke_2",
            "stroke_3"
        ]
    ]
    .mean()
    .reset_index()
)

stroke_summary = stroke_summary.rename(
    columns={
        "true_label": "character"
    }
)

# -------------------------
# MERGE
# -------------------------

profiles = stroke_summary.merge(
    coop.rename(
        columns={
            "true_label":
            "character"
        }
    ),
    on="character"
)

profiles = profiles.merge(
    synergy,
    on="character"
)

# -------------------------
# DOMINANCE SCORE
# -------------------------

def dominance(row):

    vals = [
        row["stroke_1"],
        row["stroke_2"],
        row["stroke_3"]
    ]

    vals = [v for v in vals if v > 0]

    if len(vals) == 0:
        return 0

    return max(vals) / sum(vals)

profiles["dominance_score"] = (
    profiles.apply(
        dominance,
        axis=1
    )
)

# -------------------------
# CHARACTER TYPE
# -------------------------

types = []

for _, row in profiles.iterrows():

    if row["synergy"] > 0.5:

        types.append(
            "Cooperative"
        )

    elif row["dominance_score"] > 0.8:

        types.append(
            "Dominant Stroke"
        )

    elif row["synergy"] < 0:

        types.append(
            "Redundant"
        )

    else:

        types.append(
            "Mixed"
        )

profiles["character_type"] = types

# -------------------------
# SAVE
# -------------------------

profiles = profiles.sort_values(
    "synergy",
    ascending=False
)

profiles.to_csv(
    "character_profiles.csv",
    index=False
)

print("\nTOP COOPERATIVE")
print("="*50)

print(
    profiles[
        [
            "character",
            "synergy",
            "cooperation_index",
            "dominance_score",
            "character_type"
        ]
    ].head(15)
)

print("\nMOST DOMINANT")
print("="*50)

print(
    profiles.sort_values(
        "dominance_score",
        ascending=False
    )[
        [
            "character",
            "synergy",
            "cooperation_index",
            "dominance_score",
            "character_type"
        ]
    ].head(15)
)

print("\nREDUNDANT")
print("="*50)

print(
    profiles[
        profiles["synergy"] < 0
    ][
        [
            "character",
            "synergy",
            "cooperation_index",
            "dominance_score",
            "character_type"
        ]
    ]
)

print(
    "\nSaved character_profiles.csv"
)
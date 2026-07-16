import pandas as pd

profiles = pd.read_csv("character_profiles.csv")
removal = pd.read_csv("character_removal_summary.csv")

# Keep every character from the profile table.  The removal analysis can be
# missing characters when only a subset appears in the current test split, so a
# left merge prevents the app taxonomy from losing valid model classes.
df = profiles.merge(
    removal[["character", "ratio"]],
    on="character",
    how="left",
)

categories = []

for _, row in df.iterrows():
    synergy = row["synergy"]
    ratio = row["ratio"]
    dominance = row["dominance_score"]

    if synergy < 0:
        category = "Interference"
    elif dominance > 0.80:
        category = "Dominant Stroke"
    elif synergy > 0.50 and (pd.isna(ratio) or ratio < 2):
        category = "Balanced Cooperation"
    elif synergy > 0.50 and ratio >= 2:
        category = "Asymmetric Cooperation"
    else:
        category = "Mixed"

    categories.append(category)

df["category"] = categories

print("\nCATEGORY COUNTS")
print("=" * 50)
print(df["category"].value_counts())

print("\nBALANCED COOPERATION")
print("=" * 50)
print(df[df["category"] == "Balanced Cooperation"][["character", "synergy", "ratio"]])

print("\nASYMMETRIC COOPERATION")
print("=" * 50)
print(df[df["category"] == "Asymmetric Cooperation"][["character", "synergy", "ratio"]])

print("\nDOMINANT STROKE")
print("=" * 50)
print(df[df["category"] == "Dominant Stroke"][["character", "dominance_score"]])

print("\nINTERFERENCE")
print("=" * 50)
print(df[df["category"] == "Interference"][["character", "synergy"]])

df.to_csv("character_taxonomy.csv", index=False)

print("\nSaved character_taxonomy.csv")

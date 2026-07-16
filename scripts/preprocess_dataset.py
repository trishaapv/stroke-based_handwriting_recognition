from pathlib import Path
import numpy as np

from sklearn.preprocessing import LabelEncoder

from parser import parse_uji_file
from features import extract_stroke_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

all_samples = []

for file in DATA_DIR.iterdir():

    if file.name.startswith("UJIpenchars"):

        all_samples.extend(
            parse_uji_file(file)
        )

X = []
y = []

MAX_STROKES = 3
FEATURE_DIM = 15

X = []
y = []
mask = []          # NEW: boolean array, True = real stroke, False = padding

for sample in all_samples:
    matrix = np.zeros((MAX_STROKES, FEATURE_DIM))
    stroke_mask = np.zeros(MAX_STROKES, dtype=bool)

    for i, stroke in enumerate(sample["strokes"]):
        matrix[i] = extract_stroke_features(stroke)
        stroke_mask[i] = True

    X.append(matrix)
    y.append(sample["label"])
    mask.append(stroke_mask)

X = np.array(X)
mask = np.array(mask)          # shape (N, 3), NEW

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("X shape:", X.shape)
print("mask shape:", mask.shape)

np.save("X.npy", X)
np.save("y.npy", y)
np.save("mask.npy", mask)      # NEW â€” the ground-truth record of what's real
np.save("classes.npy", encoder.classes_)

print("Saved dataset.")


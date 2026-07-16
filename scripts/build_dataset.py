from pathlib import Path

import numpy as np

from sklearn.preprocessing import LabelEncoder

from parser import parse_uji_file
from feature_extractor import stroke_features

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

PROCESSED = BASE_DIR / "processed"

PROCESSED.mkdir(exist_ok=True)

MAX_STROKES = 3

samples = []

for file in sorted(DATA_DIR.iterdir()):

    if file.name.startswith("UJIpenchars"):

        samples.extend(
            parse_uji_file(str(file))
        )

X = []
y = []

for sample in samples:

    tensor = np.zeros(
        (MAX_STROKES, 15),
        dtype=np.float32
    )

    for i, stroke in enumerate(
        sample["strokes"][:MAX_STROKES]
    ):

        tensor[i] = stroke_features(
            stroke
        )

    X.append(tensor)

    y.append(sample["label"])

X = np.array(X)

encoder = LabelEncoder()

y = encoder.fit_transform(y)

np.save(
    PROCESSED / "X.npy",
    X
)

np.save(
    PROCESSED / "y.npy",
    y
)

np.save(
    PROCESSED / "classes.npy",
    encoder.classes_
)

print("X shape:", X.shape)
print("y shape:", y.shape)
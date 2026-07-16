# dataset_builder.py

import numpy as np
from features import extract_stroke_features


def build_dataset(samples):

    X = []
    y = []

    for sample in samples:

        stroke_features = []

        for stroke in sample["strokes"]:

            feat = extract_stroke_features(stroke)

            stroke_features.append(feat)

        X.append(np.array(stroke_features))

        y.append(sample["label"])

    return X, y
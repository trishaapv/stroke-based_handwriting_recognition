from pathlib import Path

import numpy as np


class DataManager:

    def __init__(self):
        root = Path(__file__).resolve().parent.parent
        processed = root / "processed"

        # Feature matrices are kept for dashboard display; the final point model predicts from parsed raw strokes.
        self.X = np.load(processed / "X_norm.npy")
        self.y = np.load(processed / "y.npy")
        self.mask = np.load(processed / "mask.npy")
        self.classes = np.load(processed / "classes.npy", allow_pickle=True)
        self.labels = self.classes[self.y]

    def total_samples(self):
        return len(self.X)

    def get_sample(self, idx):
        return self.X[idx]

    def get_mask(self, idx):
        return self.mask[idx]

    def get_label(self, idx):
        return str(self.labels[idx])

    def get_label_index(self, idx):
        return int(self.y[idx])

    def get_class(self, idx):
        return str(self.classes[self.y[idx]])

    def get_classes(self):
        return self.classes

    def unique_labels(self):
        return sorted(np.unique(self.labels))

    def indices_for_label(self, label):
        if label == "All":
            return list(range(len(self.X)))

        return np.where(self.labels == label)[0].tolist()

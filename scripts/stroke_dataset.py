# stroke_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


class StrokeDataset(Dataset):

    def __init__(self, X_path, y_path):

        self.X = np.load(X_path)
        self.y = np.load(y_path)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        x = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        )

        y = torch.tensor(
            self.y[idx],
            dtype=torch.long
        )

        return x, y
from pathlib import Path

import numpy as np
import torch

from scripts.point_sequence_dataset import sample_to_sequence
from scripts.point_sequence_transformer import PointSequenceTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_POINTS = 96


class Predictor:

    def __init__(self):
        root = Path(__file__).resolve().parent.parent
        processed = root / "processed"

        self.classes = np.load(processed / "classes.npy", allow_pickle=True)
        self.model = PointSequenceTransformer(
            input_dim=5,
            embed_dim=128,
            num_heads=8,
            num_layers=4,
            num_classes=len(self.classes),
            max_points=MAX_POINTS,
            dropout=0.20,
        ).to(DEVICE)

        self.model.load_state_dict(
            torch.load(root / "best_point_model.pt", map_location=DEVICE)
        )
        self.model.eval()

        print("Point-sequence predictor loaded")

    def _prepare(self, parsed_sample):
        features, mask = sample_to_sequence(
            parsed_sample,
            max_points=MAX_POINTS,
            augment=False,
        )

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(DEVICE)

        return x, mask_tensor

    def _without_strokes(self, parsed_sample, removed_stroke_indices):
        removed = set(removed_stroke_indices)
        strokes = [
            stroke
            for idx, stroke in enumerate(parsed_sample["strokes"])
            if idx not in removed
        ]

        modified = dict(parsed_sample)
        modified["strokes"] = strokes
        modified["num_strokes"] = len(strokes)

        return modified

    def predict(self, parsed_sample):
        x, mask = self._prepare(parsed_sample)

        with torch.no_grad():
            logits = self.model(x, mask)
            probs = torch.softmax(logits, dim=1)[0]

        confidence, pred = torch.max(probs, dim=0)
        top_probs, top_idx = torch.topk(probs, 5)

        top5 = []
        for p, idx in zip(top_probs, top_idx):
            top5.append((str(self.classes[idx.item()]), float(p)))

        return str(self.classes[pred.item()]), float(confidence), top5

    def predict_without_strokes(self, parsed_sample, removed_stroke_indices):
        modified = self._without_strokes(parsed_sample, removed_stroke_indices)
        if len(modified["strokes"]) == 0:
            return "-", 0.0, []

        return self.predict(modified)

    def predict_proba(self, parsed_sample):
        x, mask = self._prepare(parsed_sample)

        with torch.no_grad():
            logits = self.model(x, mask)
            probs = torch.softmax(logits, dim=1)[0]

        return probs.cpu().numpy()
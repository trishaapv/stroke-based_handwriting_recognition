import itertools
from math import factorial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from point_sequence_dataset import load_uji_samples, sample_to_sequence
from point_sequence_transformer import PointSequenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "best_point_model.pt"
SEED = 42
MAX_POINTS = 96
DEVICE = torch.device("cpu")


def load_point_model(num_classes, device=DEVICE):
    model = PointSequenceTransformer(
        input_dim=5,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        num_classes=num_classes,
        max_points=MAX_POINTS,
        dropout=0.20,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_point_test_split():
    samples = load_uji_samples(DATA_DIR)
    classes = np.array(sorted({sample["label"] for sample in samples}))
    label_to_index = {label: index for index, label in enumerate(classes)}
    labels = np.array([label_to_index[sample["label"]] for sample in samples])
    indices = np.arange(len(samples))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=labels[temp_idx],
    )

    return samples, classes, label_to_index, labels, test_idx


def coalition_sample(sample, coalition):
    modified = dict(sample)
    modified["strokes"] = [
        stroke for idx, stroke in enumerate(sample["strokes"])
        if idx in coalition
    ]
    modified["num_strokes"] = len(modified["strokes"])
    return modified


def empty_sequence():
    features = np.zeros((MAX_POINTS, 5), dtype=np.float32)
    mask = np.ones(MAX_POINTS, dtype=bool)
    return features, mask


def sample_to_model_input(sample):
    if len(sample["strokes"]) == 0:
        features, mask = empty_sequence()
    else:
        features, mask = sample_to_sequence(sample, max_points=MAX_POINTS, augment=False)

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(DEVICE)
    return x, m


def payoff(model, sample, target_class, device=DEVICE):
    x, mask = sample_to_model_input(sample)
    x = x.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits = model(x, mask)
        probs = F.softmax(logits, dim=1)

    return probs[0, target_class].item()


def predict(model, sample, device=DEVICE):
    x, mask = sample_to_model_input(sample)
    x = x.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits = model(x, mask)
        probs = F.softmax(logits, dim=1)[0]

    confidence, pred = torch.max(probs, dim=0)
    return int(pred.item()), float(confidence.item())


def shapley_values(model, sample, target_class, device=DEVICE):
    n_players = len(sample["strokes"])
    players = list(range(n_players))
    shapley = np.zeros(3, dtype=float)

    if n_players == 0:
        return shapley

    n_fact = factorial(n_players)

    for player in players:
        others = [p for p in players if p != player]

        for r in range(len(others) + 1):
            for coalition in itertools.combinations(others, r):
                coalition = set(coalition)
                weight = (
                    factorial(len(coalition))
                    * factorial(n_players - len(coalition) - 1)
                ) / n_fact

                without_value = payoff(
                    model,
                    coalition_sample(sample, coalition),
                    target_class,
                    device,
                )
                with_value = payoff(
                    model,
                    coalition_sample(sample, coalition | {player}),
                    target_class,
                    device,
                )
                shapley[player] += weight * (with_value - without_value)

    return shapley


def stroke_count(sample):
    return len(sample["strokes"])
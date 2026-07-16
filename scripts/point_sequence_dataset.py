import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from parser import parse_uji_file
except ImportError:
    from scripts.parser import parse_uji_file


class PointSequenceDataset(Dataset):

    def __init__(self, samples, label_to_index, max_points=96, augment=False, seed=42):
        self.samples = samples
        self.label_to_index = label_to_index
        self.max_points = max_points
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features, mask = sample_to_sequence(
            sample,
            max_points=self.max_points,
            augment=self.augment,
            rng=self.rng,
        )
        label = self.label_to_index[sample["label"]]

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(label, dtype=torch.long),
        )


def load_uji_samples(data_dir):
    data_dir = Path(data_dir)
    samples = []

    for file in sorted(data_dir.iterdir()):
        if file.name.startswith("UJIpenchars"):
            samples.extend(parse_uji_file(file))

    return samples


def sample_to_sequence(sample, max_points=96, augment=False, rng=None):
    points, pen_flags = flatten_strokes(sample["strokes"])
    points = normalize_points(points)

    if augment:
        points = augment_points(points, rng)

    points, pen_flags = resample_sequence(points, pen_flags, max_points)

    dxdy = np.zeros_like(points)
    dxdy[1:] = points[1:] - points[:-1]

    features = np.concatenate(
        [points, dxdy, pen_flags[:, None]],
        axis=1,
    )

    mask = np.ones(max_points, dtype=bool)

    return features.astype(np.float32), mask


def flatten_strokes(strokes):
    points = []
    pen_flags = []

    for stroke in strokes:
        for point_index, (x, y) in enumerate(stroke):
            points.append((float(x), float(y)))
            pen_flags.append(1.0 if point_index > 0 else 0.0)

    return np.array(points, dtype=np.float32), np.array(pen_flags, dtype=np.float32)


def normalize_points(points):
    points = points.copy()
    center = points.mean(axis=0, keepdims=True)
    points = points - center

    scale = np.max(np.ptp(points, axis=0))
    if scale < 1e-6:
        scale = 1.0

    return points / scale


def augment_points(points, rng):
    if rng is None:
        rng = np.random.default_rng()

    angle = rng.normal(0.0, math.radians(6.0))
    scale = rng.normal(1.0, 0.05)
    shift = rng.normal(0.0, 0.025, size=(1, 2))
    noise = rng.normal(0.0, 0.01, size=points.shape)

    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float32,
    )

    return (points @ rotation.T) * scale + shift + noise


def resample_sequence(points, pen_flags, target_len):
    if len(points) == target_len:
        return points, pen_flags

    if len(points) < 2:
        repeated_points = np.repeat(points, target_len, axis=0)
        repeated_flags = np.repeat(pen_flags, target_len, axis=0)
        return repeated_points, repeated_flags

    distances = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])

    if cumulative[-1] < 1e-8:
        indices = np.linspace(0, len(points) - 1, target_len).round().astype(int)
        return points[indices], pen_flags[indices]

    target = np.linspace(0.0, cumulative[-1], target_len)
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])
    indices = np.searchsorted(cumulative, target, side="right") - 1
    indices = np.clip(indices, 0, len(pen_flags) - 1)

    return np.stack([x, y], axis=1), pen_flags[indices]

from pathlib import Path
import numpy as np

from parser import parse_uji_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

all_samples = []

for file in DATA_DIR.iterdir():
    if file.name.startswith("UJIpenchars"):
        all_samples.extend(parse_uji_file(file))

counts = []

for sample in all_samples:
    counts.append(len(sample["strokes"]))

counts = np.array(counts)

np.save("stroke_counts.npy", counts)

print(counts.shape)


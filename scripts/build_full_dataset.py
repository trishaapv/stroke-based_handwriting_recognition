from pathlib import Path

from parser import parse_uji_file
from dataset_builder import build_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

all_samples = []

for file in DATA_DIR.iterdir():

    if file.name.startswith("UJIpenchars"):

        print("Loading:", file.name)

        samples = parse_uji_file(file)

        all_samples.extend(samples)

print("\nTotal samples:", len(all_samples))

X, y = build_dataset(all_samples)

print("Dataset built.")
print("Total labels:", len(y))


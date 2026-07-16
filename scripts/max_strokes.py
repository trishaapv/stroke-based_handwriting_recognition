from parser import parse_uji_file
from dataset_builder import build_dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

all_samples = []

for file in DATA_DIR.iterdir():

    if file.name.startswith("UJIpenchars"):

        all_samples.extend(
            parse_uji_file(file)
        )

max_strokes = max(
    len(sample["strokes"])
    for sample in all_samples
)

print("Maximum strokes:", max_strokes)

stroke_counts = {}

for sample in all_samples:

    n = len(sample["strokes"])

    stroke_counts[n] = (
        stroke_counts.get(n, 0) + 1
    )

print(stroke_counts)


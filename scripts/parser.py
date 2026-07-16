# parser.py

import re
from pathlib import Path


def parse_uji_file(filepath):
    """
    Parse a UJI Pen Characters file.

    Returns:
    --------
    samples : list

    Example:

    [
        {
            "label": "a",
            "strokes": [
                [(557,844), (550,803), ...],
                [(...), (...)]
            ]
        },
        ...
    ]
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{filepath}"
        )

    samples = []

    current_label = None
    current_strokes = []
    current_stroke = []

    with open(filepath, "r", encoding="latin1") as f:

        for raw_line in f:

            line = raw_line.strip()

            # -----------------------------
            # New character segment
            # -----------------------------
            if line.startswith(".SEGMENT CHARACTER"):

                # save previous sample
                if current_label is not None:

                    if len(current_stroke) > 0:
                        current_strokes.append(current_stroke)

                    samples.append({
                        "label": current_label,
                        "strokes": current_strokes
                    })

                # extract label
                match = re.search(r'"([^"]+)"', line)

                if match:
                    current_label = match.group(1)
                else:
                    current_label = "UNKNOWN"

                current_strokes = []
                current_stroke = []

            # -----------------------------
            # Start stroke
            # -----------------------------
            elif line == ".PEN_DOWN":

                current_stroke = []

            # -----------------------------
            # End stroke
            # -----------------------------
            elif line == ".PEN_UP":

                if len(current_stroke) > 0:
                    current_strokes.append(current_stroke)

                current_stroke = []

            # -----------------------------
            # Coordinate line
            # -----------------------------
            else:

                parts = line.split()

                if len(parts) == 2:

                    try:
                        x = int(parts[0])
                        y = int(parts[1])

                        current_stroke.append((x, y))

                    except ValueError:
                        pass

    # save final sample
    if current_label is not None:

        if len(current_stroke) > 0:
            current_strokes.append(current_stroke)

        samples.append({
            "label": current_label,
            "strokes": current_strokes
        })

    return samples


def dataset_statistics(samples):

    total_chars = len(samples)

    total_strokes = sum(
        len(sample["strokes"])
        for sample in samples
    )

    avg_strokes = total_strokes / total_chars

    labels = sorted(
        list(
            set(sample["label"] for sample in samples)
        )
    )

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Characters       : {total_chars}")
    print(f"Total strokes    : {total_strokes}")
    print(f"Avg strokes/char : {avg_strokes:.2f}")
    print(f"Unique labels    : {len(labels)}")
    print(labels[:20])
    print("=" * 50)


if __name__ == "__main__":

    path = input("Enter dataset file path: ")

    samples = parse_uji_file(path)

    dataset_statistics(samples)

    print("\nFirst sample:\n")
    print("Label:", samples[0]["label"])
    print("Strokes:", len(samples[0]["strokes"]))
    print(
        "Points in first stroke:",
        len(samples[0]["strokes"][0])
    )

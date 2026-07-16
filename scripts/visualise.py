import numpy as np
import matplotlib.pyplot as plt


def plot_character(sample):

    plt.figure(figsize=(6, 6))

    for i, stroke in enumerate(sample["strokes"]):

        x = [p[0] for p in stroke]
        y = [p[1] for p in stroke]

        plt.plot(
            x,
            -np.array(y),
            linewidth=2,
            label=f"Stroke {i+1}"
        )

        plt.scatter(
            x[0],
            -y[0],
            s=50
        )

    plt.title(
        f"Character: {sample['label']}"
    )

    plt.legend()

    plt.axis("equal")

    plt.show()
import matplotlib.pyplot as plt
import io

def draw_character(sample):

    fig, ax = plt.subplots(figsize=(5,5))

    colors = [
        "#d62728",
        "#1f77b4",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e"
    ]

    for i, stroke in enumerate(sample["strokes"]):

        x = stroke[:,0]
        y = -stroke[:,1]

        ax.plot(
            x,
            y,
            linewidth=4,
            color=colors[i % len(colors)],
            label=f"Stroke {i+1}"
        )

        # mark stroke start
        ax.scatter(
            x[0],
            y[0],
            s=50,
            color="black"
        )

        ax.text(
            x[0],
            y[0],
            str(i+1),
            fontsize=10,
            weight="bold"
        )

    ax.axis("equal")
    ax.axis("off")
    ax.legend()

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()

    buffer.seek(0)

    return buffer

def draw_single_stroke(sample, stroke_id):

    fig, ax = plt.subplots(figsize=(4,4))

    stroke = sample["strokes"][stroke_id]

    x = stroke[:,0]
    y = -stroke[:,1]

    ax.plot(
        x,
        y,
        linewidth=4,
        color="#d62728"
    )

    ax.scatter(
        x[0],
        y[0],
        s=60,
        color="black"
    )

    ax.axis("equal")
    ax.axis("off")

    buf = io.BytesIO()

    plt.savefig(
        buf,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    buf.seek(0)

    return buf


def draw_order(sample):

    fig, ax = plt.subplots(figsize=(5,5))

    colors = [
        "#d62728",
        "#1f77b4",
        "#2ca02c"
    ]

    for i, stroke in enumerate(sample["strokes"]):

        x = stroke[:,0]
        y = -stroke[:,1]

        ax.plot(
            x,
            y,
            linewidth=4,
            color=colors[i]
        )

        ax.scatter(
            x[0],
            y[0],
            s=80,
            color="black"
        )

        ax.text(
            x[0],
            y[0],
            str(i+1),
            fontsize=12,
            weight="bold",
            color="black"
        )

    ax.axis("equal")
    ax.axis("off")

    buf = io.BytesIO()

    plt.savefig(
        buf,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    buf.seek(0)

    return buf

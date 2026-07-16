import numpy as np


def stroke_features(stroke):

    points = np.array(stroke)

    x = points[:, 0]
    y = points[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)

    length = np.sum(
        np.sqrt(dx**2 + dy**2)
    )

    width = x.max() - x.min()

    height = y.max() - y.min()

    orientation = np.arctan2(
        y[-1] - y[0],
        x[-1] - x[0]
    )

    mean_speed = (
        np.mean(
            np.sqrt(dx**2 + dy**2)
        )
        if len(dx) > 0
        else 0
    )

    curvature = 0

    if len(dx) > 1:

        angles = np.arctan2(dy, dx)

        curvature = np.mean(
            np.abs(
                np.diff(angles)
            )
        )

    return np.array(
        [
            length,
            width,
            height,
            orientation,
            len(stroke),
            mean_speed,
            curvature,

            x[0],
            y[0],

            x[-1],
            y[-1],

            x.mean(),
            y.mean(),

            x.std(),
            y.std()
        ],
        dtype=np.float32
    )
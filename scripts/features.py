# features_v2.py

import numpy as np


def stroke_length(stroke):
    total = 0.0

    for i in range(1, len(stroke)):
        x1, y1 = stroke[i - 1]
        x2, y2 = stroke[i]

        total += np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    return total


def bbox(stroke):

    x = np.array([p[0] for p in stroke])
    y = np.array([p[1] for p in stroke])

    return (
        x.min(),
        x.max(),
        y.min(),
        y.max()
    )


def curvature(stroke):

    if len(stroke) < 3:
        return 0

    angles = []

    for i in range(1, len(stroke)-1):

        p1 = np.array(stroke[i-1])
        p2 = np.array(stroke[i])
        p3 = np.array(stroke[i+1])

        v1 = p2 - p1
        v2 = p3 - p2

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 == 0 or n2 == 0:
            continue

        cos_theta = np.dot(v1, v2)/(n1*n2)

        cos_theta = np.clip(
            cos_theta,
            -1,
            1
        )

        angles.append(
            np.arccos(cos_theta)
        )

    if len(angles) == 0:
        return 0

    return np.mean(angles)


def extract_stroke_features(stroke):

    x = np.array([p[0] for p in stroke])
    y = np.array([p[1] for p in stroke])

    xmin, xmax, ymin, ymax = bbox(stroke)

    width = xmax - xmin
    height = ymax - ymin

    length = stroke_length(stroke)

    duration = len(stroke)

    speeds = []

    for i in range(1, len(stroke)):

        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]

        speeds.append(
            np.sqrt(dx*dx + dy*dy)
        )

    if len(speeds) == 0:
        speeds = [0]

    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    speed_var = np.var(speeds)

    direction = np.arctan2(
        y[-1] - y[0],
        x[-1] - x[0]
    )

    endpoint_distance = np.sqrt(
        (x[-1]-x[0])**2 +
        (y[-1]-y[0])**2
    )

    aspect_ratio = width / (height + 1e-6)

    compactness = (
        length /
        (width + height + 1e-6)
    )

    centroid_x = np.mean(x)
    centroid_y = np.mean(y)

    turn = curvature(stroke)

    feature_vector = np.array([
        length,
        width,
        height,
        aspect_ratio,
        duration,
        avg_speed,
        max_speed,
        speed_var,
        direction,
        turn,
        endpoint_distance,
        compactness,
        centroid_x,
        centroid_y,
        len(stroke)
    ])

    return feature_vector
"""Canny-style edge detection.

Pipeline (Canny, 1986):

1. Smooth the image with a Gaussian to suppress noise.
2. Compute Sobel gradients (magnitude + direction).
3. Non-maximum suppression: keep a pixel only if its gradient
   magnitude is a local maximum *along the gradient direction*.
   This thins edges to single-pixel width.
4. Double threshold: pixels above the high threshold are "strong"
   edges; pixels between low and high are "weak" candidates.
5. Hysteresis: a weak pixel survives only if it is connected
   (8-connectivity) to a strong pixel. This keeps continuous
   contours while discarding isolated noise responses.

Thresholds here are fractions of the maximum gradient magnitude
in the (suppressed) image, so ``low=0.1, high=0.2`` means
10% / 20% of the strongest response.
"""

from __future__ import annotations

import numpy as np

from .filters import gaussian_blur, sobel_gradients


def non_maximum_suppression(
    magnitude: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Thin edges: keep only local maxima along the gradient direction.

    The gradient direction is quantised to 4 orientations
    (0, 45, 90, 135 degrees). Each pixel is compared with the two
    neighbours lying along its gradient direction; if it is not
    >= both, it is suppressed to 0.
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    h, w = magnitude.shape
    out = np.zeros_like(magnitude)

    # Convert direction to degrees in [0, 180): an edge has no
    # "sign", theta and theta+180 describe the same line.
    angle = np.degrees(direction) % 180.0

    # Pad magnitude by 1 so border pixels can be compared too.
    padded = np.pad(magnitude, 1, mode="edge")

    # Neighbour offsets (dy, dx) for the two pixels along each
    # quantised direction. Note: image y grows downward, so a
    # gradient pointing "up" in math coords points to smaller rows.
    # Using atan2(gy, gx) with gy measured downward, the neighbour
    # pair for angle theta is symmetric, so sign conventions only
    # swap the two neighbours, which does not affect the result.
    for i in range(h):
        for j in range(w):
            a = angle[i, j]
            pi, pj = i + 1, j + 1  # position inside padded array
            if a < 22.5 or a >= 157.5:
                # Gradient ~horizontal: compare left/right neighbours.
                n1 = padded[pi, pj - 1]
                n2 = padded[pi, pj + 1]
            elif a < 67.5:
                # Gradient ~45 degrees: compare diagonal neighbours.
                n1 = padded[pi - 1, pj + 1]
                n2 = padded[pi + 1, pj - 1]
            elif a < 112.5:
                # Gradient ~vertical: compare up/down neighbours.
                n1 = padded[pi - 1, pj]
                n2 = padded[pi + 1, pj]
            else:
                # Gradient ~135 degrees: other diagonal.
                n1 = padded[pi - 1, pj - 1]
                n2 = padded[pi + 1, pj + 1]
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                out[i, j] = magnitude[i, j]
    return out


def double_threshold(
    suppressed: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split suppressed magnitudes into strong / weak masks.

    ``low`` and ``high`` are fractions (0..1) of the maximum
    magnitude. Returns ``(strong, weak)`` boolean arrays.
    """
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError("require 0 <= low <= high <= 1 (fractions of max)")
    suppressed = np.asarray(suppressed, dtype=np.float64)
    peak = suppressed.max() if suppressed.size else 0.0
    # A (near-)constant image yields gradient magnitudes at
    # floating-point noise level (~1e-17). Fractional thresholds of
    # such a peak would mark every pixel as an edge, so treat any
    # peak below a tiny epsilon as "no edges at all".
    if peak <= 1e-12:
        empty = np.zeros(suppressed.shape, dtype=bool)
        return empty, empty.copy()
    low_t = low * peak
    high_t = high * peak
    strong = suppressed >= high_t
    weak = (suppressed >= low_t) & ~strong
    return strong, weak


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Keep weak pixels that are 8-connected to a strong pixel.

    Implemented as a flood fill (stack-based DFS) seeded from every
    strong pixel, walking through weak pixels. Returns a boolean
    edge map.
    """
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)
    h, w = strong.shape
    edges = strong.copy()
    visited = strong.copy()

    stack = [(int(i), int(j)) for i, j in zip(*np.nonzero(strong))]
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    while stack:
        i, j = stack.pop()
        for di, dj in neighbours:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj]:
                if weak[ni, nj]:
                    visited[ni, nj] = True
                    edges[ni, nj] = True
                    stack.append((ni, nj))
    return edges


def canny(
    image: np.ndarray,
    sigma: float = 1.0,
    low: float = 0.1,
    high: float = 0.2,
) -> np.ndarray:
    """Run the full Canny pipeline. Returns a boolean edge map."""
    image = np.asarray(image, dtype=np.float64)
    smoothed = gaussian_blur(image, sigma=sigma)
    _, _, magnitude, direction = sobel_gradients(smoothed)
    suppressed = non_maximum_suppression(magnitude, direction)
    strong, weak = double_threshold(suppressed, low, high)
    return hysteresis(strong, weak)

"""Binary thresholding: fixed threshold and Otsu's method.

Fixed thresholding:

    out = 1 where image >= t, else 0

Otsu's method (Otsu, 1979) picks the threshold automatically by
maximising the *between-class variance* of the two groups
(background / foreground) the threshold creates. Equivalently it
minimises the within-class variance. For a candidate threshold t
splitting pixels into classes C0 (<= t) and C1 (> t) with weights
w0, w1 (class probabilities) and means mu0, mu1:

    sigma_b^2(t) = w0 * w1 * (mu0 - mu1)^2

The t maximising sigma_b^2 is chosen. The search runs over a
256-bin histogram spanning the image's intensity range, and the
returned threshold is in the image's original intensity scale.
"""

from __future__ import annotations

import numpy as np


def fixed_threshold(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a boolean mask: True where ``image >= threshold``."""
    image = np.asarray(image, dtype=np.float64)
    return image >= threshold


def otsu_threshold(image: np.ndarray, bins: int = 256) -> float:
    """Compute Otsu's threshold for ``image`` (original intensity scale)."""
    image = np.asarray(image, dtype=np.float64)
    if image.size == 0:
        raise ValueError("image must not be empty")

    lo = float(image.min())
    hi = float(image.max())
    if hi == lo:
        # Constant image: no split exists; return the only value.
        return lo

    hist, bin_edges = np.histogram(image, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    # Bin centres represent the intensity of pixels in each bin.
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total = hist.sum()
    prob = hist / total

    # Cumulative weight and mean, updated incrementally per split.
    omega = np.cumsum(prob)                      # w0 for each split
    mu_cum = np.cumsum(prob * centres)           # w0 * mu0
    mu_total = mu_cum[-1]

    # Between-class variance for every split (vectorised):
    #   sigma_b^2 = (mu_total * w0 - mu_cum)^2 / (w0 * (1 - w0))
    denom = omega * (1.0 - omega)
    # Splits where one class is empty have denom 0 -> variance 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_sq = np.where(
            denom > 0,
            (mu_total * omega - mu_cum) ** 2 / np.where(denom > 0, denom, 1.0),
            0.0,
        )

    best = int(np.argmax(sigma_b_sq))
    # Threshold = upper edge of the best background bin.
    return float(bin_edges[best + 1])


def otsu_binarize(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Binarize with Otsu's method. Returns ``(mask, threshold)``."""
    threshold = otsu_threshold(image)
    return fixed_threshold(image, threshold), threshold

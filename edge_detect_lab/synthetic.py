"""Synthetic demo images generated in code (no external files).

Every generator returns a float64 array of shape ``(size, size)``
with intensities in [0, 1]. Two intensity levels are used for the
shapes: background 0.2, foreground 0.8 — a clean bimodal image,
which makes edge responses and Otsu's method easy to verify.

Geometry (documented so tests can rely on it):

- ``square``: filled square covering rows/cols
  ``[size//4, 3*size//4)``.
- ``circle``: filled circle centred in the image, radius ``size//4``.
- ``checkerboard``: 8x8 alternating blocks.
- ``step``: left half 0.2, right half 0.8 (vertical step edge at
  column ``size//2``).
- ``gradient``: horizontal linear ramp from 0.0 (left) to 1.0 (right).
"""

from __future__ import annotations

import numpy as np

BACKGROUND = 0.2
FOREGROUND = 0.8

IMAGE_NAMES = ("square", "circle", "checkerboard", "step", "gradient")


def make_square(size: int = 128) -> np.ndarray:
    image = np.full((size, size), BACKGROUND, dtype=np.float64)
    lo = size // 4
    hi = 3 * size // 4
    image[lo:hi, lo:hi] = FOREGROUND
    return image


def make_circle(size: int = 128) -> np.ndarray:
    image = np.full((size, size), BACKGROUND, dtype=np.float64)
    centre = (size - 1) / 2.0
    radius = size / 4.0
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (yy - centre) ** 2 + (xx - centre) ** 2 <= radius ** 2
    image[mask] = FOREGROUND
    return image


def make_checkerboard(size: int = 128, blocks: int = 8) -> np.ndarray:
    block = max(1, size // blocks)
    yy, xx = np.mgrid[0:size, 0:size]
    parity = (yy // block + xx // block) % 2
    image = np.where(parity == 0, BACKGROUND, FOREGROUND).astype(np.float64)
    return image


def make_step(size: int = 128) -> np.ndarray:
    image = np.full((size, size), BACKGROUND, dtype=np.float64)
    image[:, size // 2:] = FOREGROUND
    return image


def make_gradient(size: int = 128) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return np.tile(ramp, (size, 1))


_GENERATORS = {
    "square": make_square,
    "circle": make_circle,
    "checkerboard": make_checkerboard,
    "step": make_step,
    "gradient": make_gradient,
}


def make_image(name: str, size: int = 128) -> np.ndarray:
    """Generate the synthetic image called ``name`` at ``size`` x ``size``."""
    if name not in _GENERATORS:
        raise ValueError(f"unknown image {name!r}; expected one of {IMAGE_NAMES}")
    if size < 8:
        raise ValueError("size must be at least 8")
    return _GENERATORS[name](size)

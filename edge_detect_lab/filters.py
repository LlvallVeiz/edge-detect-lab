"""Gaussian blur and Sobel gradients.

Gaussian kernel (2D):

    G(x, y) = exp( -(x^2 + y^2) / (2 * sigma^2) )

The kernel is sampled on a discrete grid and normalised so its
weights sum to 1 (a normalised kernel preserves the average
brightness of the image).

The 2D Gaussian is *separable*: G(x, y) = G(x) * G(y). Blurring
with the full 2D kernel therefore equals a horizontal 1D blur
followed by a vertical 1D blur, which is cheaper (2k taps instead
of k^2). :func:`gaussian_blur` uses the separable form.

Sobel kernels (correlation form):

    Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
          [-2, 0, 2],           [ 0,  0,  0],
          [-1, 0, 1]]           [ 1,  2,  1]]

Gx responds to vertical edges (intensity changing along x),
Gy to horizontal edges. Gradient magnitude and direction:

    |G|   = sqrt(Gx^2 + Gy^2)
    theta = atan2(Gy, Gx)
"""

from __future__ import annotations

import math

import numpy as np

from .convolution import convolve2d

SOBEL_X = np.array(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
)

SOBEL_Y = np.array(
    [[-1.0, -2.0, -1.0],
     [0.0, 0.0, 0.0],
     [1.0, 2.0, 1.0]]
)


def _kernel_radius(sigma: float, kernel_size: int | None) -> int:
    if kernel_size is not None:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return kernel_size // 2
    # Cover +/-3 sigma: beyond that the Gaussian is < 1% of its peak.
    return max(1, int(math.ceil(3.0 * sigma)))


def gaussian_kernel_1d(sigma: float, kernel_size: int | None = None) -> np.ndarray:
    """Return a normalised 1D Gaussian kernel for the given ``sigma``."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = _kernel_radius(sigma, kernel_size)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_kernel(sigma: float, kernel_size: int | None = None) -> np.ndarray:
    """Return a normalised 2D Gaussian kernel for the given ``sigma``.

    Built as the outer product of the 1D kernel with itself, which
    is exactly the sampled 2D Gaussian because it is separable.
    """
    k1d = gaussian_kernel_1d(sigma, kernel_size)
    kernel = np.outer(k1d, k1d)
    kernel /= kernel.sum()
    return kernel


def gaussian_blur(
    image: np.ndarray,
    sigma: float = 1.0,
    kernel_size: int | None = None,
    padding: str = "edge",
) -> np.ndarray:
    """Blur ``image`` with a Gaussian of width ``sigma`` (separable)."""
    image = np.asarray(image, dtype=np.float64)
    k1d = gaussian_kernel_1d(sigma, kernel_size)
    # Horizontal pass (1 x k), then vertical pass (k x 1).
    blurred = convolve2d(image, k1d.reshape(1, -1), padding=padding, flip=False)
    blurred = convolve2d(blurred, k1d.reshape(-1, 1), padding=padding, flip=False)
    return blurred


def sobel_gradients(
    image: np.ndarray,
    padding: str = "edge",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel gradients of ``image``.

    Returns
    -------
    gx, gy, magnitude, direction
        ``gx``/``gy`` are the horizontal/vertical derivative
        responses, ``magnitude = sqrt(gx^2 + gy^2)`` and
        ``direction = atan2(gy, gx)`` in radians, in [-pi, pi].
    """
    image = np.asarray(image, dtype=np.float64)
    gx = convolve2d(image, SOBEL_X, padding=padding, flip=False)
    gy = convolve2d(image, SOBEL_Y, padding=padding, flip=False)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)
    return gx, gy, magnitude, direction

"""2D convolution engine implemented with numpy.

Mathematical definition (true convolution):

    (f * g)[i, j] = sum_m sum_n f[i - m, j - n] * g[m, n]

True convolution flips the kernel both horizontally and vertically
before sliding it over the image. Image-processing libraries often
implement *cross-correlation* instead (no flip) and still call it
"convolution". This module implements true convolution by default
(``flip=True``); filters in :mod:`edge_detect_lab.filters` pass
``flip=False`` because their kernels (Sobel, Gaussian) are written
in correlation form.

Padding modes (how pixels outside the image are invented):

- ``"zero"``    : pad with 0.0
- ``"edge"``    : replicate the nearest edge pixel
- ``"reflect"`` : mirror the image at the border (numpy ``reflect``)

The implementation is vectorised over the image: for each kernel
position it adds a shifted copy of the padded image scaled by the
kernel weight. That is O(kh * kw) numpy operations on whole arrays,
which is plenty fast for the small kernels used in this lab.
"""

from __future__ import annotations

import numpy as np

PADDING_MODES = ("zero", "edge", "reflect")

_NUMPY_PAD_MODE = {
    "zero": "constant",
    "edge": "edge",
    "reflect": "reflect",
}


def _pad(image: np.ndarray, pad_h: int, pad_w: int, padding: str) -> np.ndarray:
    if padding not in _NUMPY_PAD_MODE:
        raise ValueError(
            f"unknown padding mode {padding!r}; expected one of {PADDING_MODES}"
        )
    if pad_h == 0 and pad_w == 0:
        return image
    mode = _NUMPY_PAD_MODE[padding]
    if mode == "constant":
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                      mode="constant", constant_values=0.0)
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding: str = "edge",
    flip: bool = True,
) -> np.ndarray:
    """Convolve a 2D grayscale ``image`` with a 2D ``kernel``.

    Parameters
    ----------
    image:
        2D array (converted to float64 internally).
    kernel:
        2D array; odd dimensions are typical but even sizes work too.
    padding:
        One of ``"zero"``, ``"edge"``, ``"reflect"``.
    flip:
        If True (default), flip the kernel in both axes first, i.e.
        true mathematical convolution. If False, compute
        cross-correlation (what most image filters expect).

    Returns
    -------
    np.ndarray
        Float64 array with the same shape as ``image`` ("same" output).
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D, got shape {kernel.shape}")
    if kernel.size == 0:
        raise ValueError("kernel must not be empty")

    if flip:
        kernel = kernel[::-1, ::-1]

    kh, kw = kernel.shape
    h, w = image.shape

    # Padding needed so every output pixel sees a full kernel window.
    # For even kernels the extra pixel goes on the bottom/right.
    pad_top = kh // 2
    pad_bottom = kh - 1 - pad_top
    pad_left = kw // 2
    pad_right = kw - 1 - pad_left

    # np.pad with asymmetric widths: pad manually via the helper that
    # pads symmetrically, then crop if asymmetric. Simpler: use
    # np.pad directly here with per-side widths.
    if padding not in _NUMPY_PAD_MODE:
        raise ValueError(
            f"unknown padding mode {padding!r}; expected one of {PADDING_MODES}"
        )
    mode = _NUMPY_PAD_MODE[padding]
    pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    if mode == "constant":
        padded = np.pad(image, pad_width, mode="constant", constant_values=0.0)
    else:
        padded = np.pad(image, pad_width, mode=mode)

    out = np.zeros((h, w), dtype=np.float64)
    # For each kernel tap, add the correspondingly shifted window of
    # the padded image. This is exactly the sliding dot product.
    for i in range(kh):
        for j in range(kw):
            weight = kernel[i, j]
            if weight == 0.0:
                continue
            out += weight * padded[i:i + h, j:j + w]
    return out

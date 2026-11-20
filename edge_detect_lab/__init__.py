"""edge-detect-lab: a from-scratch image-processing lab.

Built with numpy + matplotlib only (no OpenCV, no PIL).

Public API re-exported here for convenience.
"""

from .convolution import convolve2d
from .filters import (
    gaussian_kernel,
    gaussian_kernel_1d,
    gaussian_blur,
    sobel_gradients,
)
from .canny import canny, non_maximum_suppression, double_threshold, hysteresis
from .threshold import fixed_threshold, otsu_threshold, otsu_binarize
from .synthetic import make_image, IMAGE_NAMES

__all__ = [
    "convolve2d",
    "gaussian_kernel",
    "gaussian_kernel_1d",
    "gaussian_blur",
    "sobel_gradients",
    "canny",
    "non_maximum_suppression",
    "double_threshold",
    "hysteresis",
    "fixed_threshold",
    "otsu_threshold",
    "otsu_binarize",
    "make_image",
    "IMAGE_NAMES",
]

__version__ = "1.0.0"

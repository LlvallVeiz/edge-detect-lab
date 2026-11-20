"""Matplotlib plotting helpers (headless, Agg backend).

Only numpy + matplotlib are used — no PIL/OpenCV image IO. All
figures are saved as PNG files.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: safe on headless machines
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _show(ax, image: np.ndarray, title: str, cmap: str = "gray") -> None:
    ax.imshow(np.asarray(image, dtype=np.float64), cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")


def save_side_by_side(
    before: np.ndarray,
    after: np.ndarray,
    before_title: str,
    after_title: str,
    out_path: str | Path,
    after_cmap: str = "gray",
) -> Path:
    """Save a 1x2 before/after figure to ``out_path`` (PNG)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _show(axes[0], before, before_title)
    # For gradient/magnitude views the natural range is not [0, 1];
    # imshow with vmin/vmax fixed would clip them. Detect that case:
    # if the data exceeds [0, 1], let matplotlib autoscale instead.
    after_arr = np.asarray(after, dtype=np.float64)
    if after_arr.min() < 0.0 or after_arr.max() > 1.0:
        axes[1].imshow(after_arr, cmap=after_cmap)
        axes[1].set_title(after_title)
        axes[1].axis("off")
    else:
        _show(axes[1], after_arr, after_title, cmap=after_cmap)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_compare_grid(
    panels: list[tuple[str, np.ndarray]],
    out_path: str | Path,
    suptitle: str = "edge-detect-lab: method comparison",
) -> Path:
    """Save a grid figure with one panel per (title, image) pair."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(panels)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (title, image) in zip(axes_flat, panels):
        arr = np.asarray(image, dtype=np.float64)
        if arr.min() < 0.0 or arr.max() > 1.0:
            ax.imshow(arr, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        else:
            _show(ax, arr, title)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

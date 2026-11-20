"""Command-line interface for edge-detect-lab.

Subcommands (each works on a generated synthetic image and saves a
matplotlib PNG):

- ``blur``      Gaussian blur
- ``sobel``     Sobel gradient (magnitude / gx / gy / direction)
- ``canny``     Canny edge detection
- ``threshold`` Binary thresholding (fixed or Otsu)
- ``compare``   All methods side by side in one grid figure

Common options: ``--image square|circle|checkerboard|step|gradient``,
``--size N``, ``--out PATH``.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .canny import canny
from .filters import gaussian_blur, sobel_gradients
from .plotting import save_compare_grid, save_side_by_side
from .synthetic import IMAGE_NAMES, make_image
from .threshold import fixed_threshold, otsu_binarize


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image", choices=IMAGE_NAMES, default="square",
        help="synthetic demo image to generate (default: square)",
    )
    parser.add_argument(
        "--size", type=int, default=128,
        help="image size in pixels (square image, default: 128)",
    )
    parser.add_argument(
        "--out", required=True,
        help="output PNG path",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-detect-lab",
        description="From-scratch image-processing lab (numpy + matplotlib only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_blur = sub.add_parser("blur", help="Gaussian blur")
    _add_common(p_blur)
    p_blur.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian sigma (default: 2.0)")

    p_sobel = sub.add_parser("sobel", help="Sobel gradient")
    _add_common(p_sobel)
    p_sobel.add_argument(
        "--mode", choices=("magnitude", "gx", "gy", "direction"),
        default="magnitude", help="which gradient view to save (default: magnitude)",
    )

    p_canny = sub.add_parser("canny", help="Canny edge detection")
    _add_common(p_canny)
    p_canny.add_argument("--sigma", type=float, default=1.0,
                         help="smoothing sigma (default: 1.0)")
    p_canny.add_argument("--low", type=float, default=0.1,
                         help="low threshold, fraction of max gradient (default: 0.1)")
    p_canny.add_argument("--high", type=float, default=0.2,
                         help="high threshold, fraction of max gradient (default: 0.2)")

    p_thr = sub.add_parser("threshold", help="binary thresholding")
    _add_common(p_thr)
    p_thr.add_argument("--method", choices=("fixed", "otsu"), default="otsu",
                       help="thresholding method (default: otsu)")
    p_thr.add_argument("--thresh", type=float, default=0.5,
                       help="fixed threshold value (default: 0.5)")

    p_cmp = sub.add_parser("compare", help="all methods in one grid figure")
    _add_common(p_cmp)
    p_cmp.add_argument("--sigma", type=float, default=1.5,
                       help="Gaussian sigma for blur/Canny (default: 1.5)")

    return parser


def _stats_line(label: str, arr: np.ndarray) -> str:
    return (f"{label}: shape={arr.shape} min={arr.min():.4f} "
            f"max={arr.max():.4f} mean={arr.mean():.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    image = make_image(args.image, size=args.size)
    print(f"edge-detect-lab: command={args.command} image={args.image} size={args.size}")
    print(_stats_line("input", image))

    if args.command == "blur":
        result = gaussian_blur(image, sigma=args.sigma)
        save_side_by_side(image, result, "original",
                          f"gaussian blur (sigma={args.sigma})", args.out)
        print(_stats_line("blurred", result))

    elif args.command == "sobel":
        gx, gy, magnitude, direction = sobel_gradients(image)
        views = {"magnitude": magnitude, "gx": gx, "gy": gy, "direction": direction}
        result = views[args.mode]
        # Normalise magnitude to [0, 1] for display; keep gx/gy/direction raw.
        display = result
        if args.mode == "magnitude" and result.max() > 0:
            display = result / result.max()
        save_side_by_side(image, display, "original", f"sobel {args.mode}", args.out)
        print(_stats_line(f"sobel {args.mode}", result))

    elif args.command == "canny":
        edges = canny(image, sigma=args.sigma, low=args.low, high=args.high)
        save_side_by_side(image, edges.astype(np.float64), "original",
                          "canny edges", args.out)
        print(f"canny: edge pixels = {int(edges.sum())} / {edges.size} "
              f"({100.0 * edges.mean():.2f}%)")

    elif args.command == "threshold":
        if args.method == "otsu":
            mask, thr = otsu_binarize(image)
            print(f"otsu: chosen threshold = {thr:.4f}")
        else:
            thr = args.thresh
            mask = fixed_threshold(image, thr)
            print(f"fixed: threshold = {thr:.4f}")
        save_side_by_side(image, mask.astype(np.float64), "original",
                          f"binary ({args.method})", args.out)
        print(f"threshold: foreground pixels = {int(mask.sum())} / {mask.size} "
              f"({100.0 * mask.mean():.2f}%)")

    elif args.command == "compare":
        blurred = gaussian_blur(image, sigma=args.sigma)
        _, _, magnitude, _ = sobel_gradients(image)
        mag_norm = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
        edges = canny(image, sigma=args.sigma)
        otsu_mask, otsu_thr = otsu_binarize(image)
        fixed_mask = fixed_threshold(image, 0.5)
        panels = [
            ("original", image),
            (f"gaussian blur (sigma={args.sigma})", blurred),
            ("sobel magnitude", mag_norm),
            ("canny edges", edges.astype(np.float64)),
            (f"otsu binary (t={otsu_thr:.3f})", otsu_mask.astype(np.float64)),
            ("fixed binary (t=0.5)", fixed_mask.astype(np.float64)),
        ]
        save_compare_grid(panels, args.out,
                          suptitle=f"edge-detect-lab: {args.image} ({args.size}x{args.size})")
        print(f"compare: otsu threshold = {otsu_thr:.4f}, "
              f"canny edge pixels = {int(edges.sum())}")

    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

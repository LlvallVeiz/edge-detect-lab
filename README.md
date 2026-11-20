# edge-detect-lab

A from-scratch image-processing lab built with **numpy + matplotlib only**
(no OpenCV, no PIL). It implements the classic edge-detection toolbox by
hand — convolution, Gaussian blur, Sobel gradients, Canny edge detection,
and thresholding (fixed + Otsu) — and demonstrates every method on synthetic
images generated in code, so no external image files are needed.

Requires Python 3.12+, numpy, matplotlib. Tests use the standard-library
`unittest` (no pytest needed). Matplotlib runs on the headless `Agg`
backend, so everything works without a display.

## What it does

| Module | Contents |
|---|---|
| `edge_detect_lab/convolution.py` | 2D convolution engine with padding modes (`zero`, `edge`, `reflect`) |
| `edge_detect_lab/filters.py` | Gaussian kernel/blur (separable), Sobel gradients |
| `edge_detect_lab/canny.py` | Canny pipeline: smoothing → Sobel → non-maximum suppression → double threshold → hysteresis |
| `edge_detect_lab/threshold.py` | Fixed thresholding and Otsu's automatic threshold |
| `edge_detect_lab/synthetic.py` | Synthetic demo images (square, circle, checkerboard, step, gradient) |
| `edge_detect_lab/plotting.py` | Headless matplotlib before/after and comparison-grid PNGs |
| `edge_detect_lab/cli.py` | `argparse` CLI with subcommands `blur`, `sobel`, `canny`, `threshold`, `compare` |

## Quick start

Run from the project root (the folder containing `edge_detect_lab/`):

```bash
# Gaussian blur of a synthetic circle
python3 -m edge_detect_lab blur --image circle --size 128 --sigma 2.0 --out blur.png

# Sobel gradient magnitude of a checkerboard
python3 -m edge_detect_lab sobel --image checkerboard --mode magnitude --out sobel.png
# other modes: --mode gx | gy | direction

# Canny edges of a square
python3 -m edge_detect_lab canny --image square --sigma 1.0 --low 0.1 --high 0.2 --out canny.png

# Binary thresholding (Otsu picks the threshold automatically)
python3 -m edge_detect_lab threshold --image square --method otsu --out otsu.png
python3 -m edge_detect_lab threshold --image square --method fixed --thresh 0.5 --out fixed.png

# Every method on one image, in a single grid figure
python3 -m edge_detect_lab compare --image circle --size 128 --out compare.png
```

Common options for every subcommand:

- `--image square|circle|checkerboard|step|gradient` — synthetic image to generate
- `--size N` — image is `N × N` pixels (default 128)
- `--out PATH` — output PNG (required); each command saves a side-by-side
  before/after plot (`compare` saves a 2×3 grid) and prints a short summary
  (image stats, edge-pixel count, chosen threshold) to stdout.

## The math

### Convolution

True 2D convolution flips the kernel before sliding it over the image:

```
(f * g)[i, j] = Σ_m Σ_n f[i − m, j − n] · g[m, n]
```

`convolve2d` implements this directly (vectorised over the image, one
shifted copy per kernel tap). Pixels outside the image are invented by the
padding mode: `zero` (0.0), `edge` (replicate border), or `reflect`
(mirror). Filters that are written in correlation form (Sobel, Gaussian)
call the engine with `flip=False`.

### Gaussian blur

```
G(x, y) = exp( −(x² + y²) / (2σ²) )
```

The kernel is sampled on a grid covering ±3σ and **normalised to sum to 1**,
so average brightness is preserved. Because the 2D Gaussian is separable
(`G(x, y) = G(x)·G(y)`), the blur runs as a horizontal 1D pass followed by a
vertical 1D pass — 2k taps instead of k².

### Sobel gradients

```
Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
      [-2, 0, 2],           [ 0,  0,  0],
      [-1, 0, 1]]           [ 1,  2,  1]]

magnitude = sqrt(Gx² + Gy²)
direction = atan2(Gy, Gx)
```

`Gx` responds to vertical edges (intensity changing along x), `Gy` to
horizontal edges.

### Canny edge detection

1. **Smooth** with a Gaussian (σ configurable) to suppress noise.
2. **Gradients** via Sobel → magnitude + direction per pixel.
3. **Non-maximum suppression**: the direction is quantised to
   0°/45°/90°/135°; a pixel survives only if its magnitude is ≥ both
   neighbours along its gradient direction. This thins edges to 1 px.
4. **Double threshold**: magnitudes ≥ `high`·max are *strong* edges,
   between `low`·max and `high`·max are *weak* candidates.
5. **Hysteresis**: a weak pixel is kept only if it is 8-connected to a
   strong pixel (flood fill), so real contours stay continuous while
   isolated noise responses are discarded.

### Thresholding and Otsu's method

Fixed thresholding: `out = 1 where image ≥ t, else 0`.

Otsu's method chooses `t` automatically by maximising the **between-class
variance** of the two groups the threshold creates. For classes C₀ (≤ t)
and C₁ (> t) with weights w₀, w₁ and means μ₀, μ₁:

```
σ_b²(t) = w₀ · w₁ · (μ₀ − μ₁)²
```

The search runs over a 256-bin histogram of the image; the returned
threshold is in the image's original intensity scale. On a bimodal image
(e.g. the synthetic square: background 0.2, foreground 0.8) it cleanly
separates the two levels.

### Synthetic images

Generated in code, intensities in [0, 1], background 0.2 / foreground 0.8:

- `square` — filled square covering rows/cols `[size//4, 3·size//4)`
- `circle` — filled circle, centre of image, radius `size/4`
- `checkerboard` — 8×8 alternating blocks
- `step` — left half 0.2, right half 0.8 (vertical step edge)
- `gradient` — horizontal linear ramp 0.0 → 1.0

## Project structure

```
edge-detect-lab/
├── README.md
├── edge_detect_lab/
│   ├── __init__.py      # public API re-exports
│   ├── __main__.py      # python -m edge_detect_lab entry point
│   ├── convolution.py   # 2D convolution engine + padding modes
│   ├── filters.py       # Gaussian blur, Sobel gradients
│   ├── canny.py         # Canny pipeline (NMS, double threshold, hysteresis)
│   ├── threshold.py     # fixed threshold, Otsu's method
│   ├── synthetic.py     # synthetic demo image generators
│   ├── plotting.py      # headless matplotlib PNG output
│   └── cli.py           # argparse CLI (blur/sobel/canny/threshold/compare)
└── tests/
    ├── test_convolution.py
    ├── test_filters.py
    ├── test_canny.py
    ├── test_threshold.py
    └── test_cli.py
```

## Running the tests

```bash
cd edge-detect-lab
python3 -m unittest discover -s tests -v
```

The suite covers: identity/known-kernel convolution (including the kernel
flip), Gaussian kernel normalisation and mean preservation, Sobel response
on a vertical step edge (strong at the edge, ~0 away), the Canny pipeline
on a synthetic square (edges land on the square boundary), each Canny stage
individually (NMS, double threshold, hysteresis), Otsu separation of a
bimodal image, and CLI smoke tests that run subcommands and check the PNG
output exists.

## Using it as a library

```python
import numpy as np
from edge_detect_lab import make_image, gaussian_blur, sobel_gradients, canny, otsu_binarize

image = make_image("square", size=128)
blurred = gaussian_blur(image, sigma=2.0)
gx, gy, magnitude, direction = sobel_gradients(image)
edges = canny(image, sigma=1.0, low=0.1, high=0.2)   # boolean edge map
mask, threshold = otsu_binarize(image)
```

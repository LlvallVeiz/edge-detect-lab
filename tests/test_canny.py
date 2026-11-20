"""Tests for the Canny pipeline."""

import unittest

import numpy as np

from edge_detect_lab.canny import (
    canny,
    double_threshold,
    hysteresis,
    non_maximum_suppression,
)
from edge_detect_lab.synthetic import make_square, make_step


class TestCannyPipeline(unittest.TestCase):
    def test_canny_detects_square_boundary(self):
        size = 64
        image = make_square(size)
        edges = canny(image, sigma=1.0, low=0.1, high=0.2)

        self.assertEqual(edges.dtype, bool)
        self.assertGreater(int(edges.sum()), 0)

        # The square occupies [size//4, 3*size//4) in both axes.
        # Its boundary is the perimeter of that block. Check that
        # most detected edge pixels lie within 2 px of the boundary.
        lo = size // 4
        hi = 3 * size // 4  # exclusive
        boundary = np.zeros((size, size), dtype=bool)
        boundary[lo:hi, lo] = True
        boundary[lo:hi, hi - 1] = True
        boundary[lo, lo:hi] = True
        boundary[hi - 1, lo:hi] = True

        # Dilate boundary by 2 px (Chebyshev) via shifts.
        near = boundary.copy()
        for _ in range(2):
            shifted = near.copy()
            shifted[1:, :] |= near[:-1, :]
            shifted[:-1, :] |= near[1:, :]
            shifted[:, 1:] |= near[:, :-1]
            shifted[:, :-1] |= near[:, 1:]
            near = shifted

        on_boundary = int((edges & near).sum())
        fraction = on_boundary / int(edges.sum())
        self.assertGreater(fraction, 0.8)

        # And the interior / far background must be edge-free.
        self.assertFalse(edges[size // 2, size // 2])
        self.assertFalse(edges[2, 2])

    def test_canny_on_step_edge(self):
        size = 64
        edges = canny(make_step(size), sigma=1.0)
        # Edge pixels should concentrate at the step columns.
        cols_with_edges = np.nonzero(edges.any(axis=0))[0]
        self.assertTrue(len(cols_with_edges) > 0)
        for col in cols_with_edges:
            self.assertIn(col, (size // 2 - 1, size // 2))

    def test_canny_on_constant_image_finds_nothing(self):
        edges = canny(np.full((32, 32), 0.5))
        self.assertEqual(int(edges.sum()), 0)


class TestCannyStages(unittest.TestCase):
    def test_non_maximum_suppression_thins_ridge(self):
        # Horizontal ridge of magnitude: row 5 peaks, rows 4/6 lower.
        magnitude = np.zeros((11, 11))
        magnitude[4, :] = 1.0
        magnitude[5, :] = 3.0
        magnitude[6, :] = 1.0
        # Gradient pointing vertically (90 deg) => compare up/down.
        direction = np.full((11, 11), np.pi / 2)
        out = non_maximum_suppression(magnitude, direction)
        np.testing.assert_allclose(out[5, :], 3.0)
        np.testing.assert_allclose(out[4, :], 0.0)
        np.testing.assert_allclose(out[6, :], 0.0)

    def test_double_threshold_masks(self):
        suppressed = np.array([[0.0, 0.05, 0.15, 0.5, 1.0]])
        strong, weak = double_threshold(suppressed, low=0.1, high=0.3)
        np.testing.assert_array_equal(strong, [[False, False, False, True, True]])
        np.testing.assert_array_equal(weak, [[False, False, True, False, False]])

    def test_hysteresis_keeps_connected_weak_only(self):
        strong = np.zeros((5, 5), dtype=bool)
        weak = np.zeros((5, 5), dtype=bool)
        strong[2, 0] = True
        weak[2, 1] = True   # connected to strong -> kept
        weak[2, 2] = True   # connected via (2,1) -> kept
        weak[0, 4] = True   # isolated -> dropped
        edges = hysteresis(strong, weak)
        self.assertTrue(edges[2, 0])
        self.assertTrue(edges[2, 1])
        self.assertTrue(edges[2, 2])
        self.assertFalse(edges[0, 4])


if __name__ == "__main__":
    unittest.main()

"""Tests for Gaussian blur and Sobel gradients."""

import unittest

import numpy as np

from edge_detect_lab.filters import (
    gaussian_blur,
    gaussian_kernel,
    gaussian_kernel_1d,
    sobel_gradients,
)
from edge_detect_lab.synthetic import make_gradient, make_step


class TestGaussian(unittest.TestCase):
    def test_kernel_sums_to_one(self):
        for sigma in (0.5, 1.0, 2.0, 3.5):
            self.assertAlmostEqual(gaussian_kernel_1d(sigma).sum(), 1.0, places=12)
            self.assertAlmostEqual(gaussian_kernel(sigma).sum(), 1.0, places=12)

    def test_kernel_is_symmetric_and_peaks_at_centre(self):
        k = gaussian_kernel_1d(1.5)
        np.testing.assert_allclose(k, k[::-1])
        self.assertEqual(int(np.argmax(k)), len(k) // 2)

    def test_blur_preserves_mean_interior(self):
        image = make_gradient(64)
        blurred = gaussian_blur(image, sigma=2.0)
        # Away from the borders, a normalised symmetric kernel keeps
        # the local average: interior means must match closely.
        interior = np.s_[10:-10, 10:-10]
        self.assertAlmostEqual(
            float(blurred[interior].mean()), float(image[interior].mean()), places=6
        )

    def test_blur_of_constant_image_is_constant(self):
        image = np.full((20, 20), 0.7)
        blurred = gaussian_blur(image, sigma=1.5)
        np.testing.assert_allclose(blurred, image, atol=1e-12)

    def test_blur_smooths_step_edge(self):
        image = make_step(64)
        blurred = gaussian_blur(image, sigma=2.0)
        # The sharp jump 0.2 -> 0.8 becomes a ramp: max adjacent
        # difference strictly decreases.
        self.assertLess(np.abs(np.diff(blurred, axis=1)).max(),
                        np.abs(np.diff(image, axis=1)).max())


class TestSobel(unittest.TestCase):
    def test_vertical_step_edge_response(self):
        size = 64
        image = make_step(size)  # vertical edge at column size//2
        gx, gy, magnitude, _ = sobel_gradients(image)

        edge_col = size // 2
        # Strong response at the two columns straddling the step.
        self.assertGreater(magnitude[:, edge_col - 1].min(), 1.0)
        self.assertGreater(magnitude[:, edge_col].min(), 1.0)
        # Near-zero away from the edge.
        self.assertLess(np.abs(magnitude[:, 5]).max(), 1e-9)
        self.assertLess(np.abs(magnitude[:, size - 5]).max(), 1e-9)
        # A vertical edge is a change along x: |gx| dominates, gy ~ 0.
        self.assertLess(np.abs(gy).max(), 1e-9)
        self.assertGreater(np.abs(gx[:, edge_col]).min(), 1.0)

    def test_constant_image_has_zero_gradient(self):
        image = np.full((16, 16), 0.5)
        gx, gy, magnitude, _ = sobel_gradients(image)
        np.testing.assert_allclose(magnitude, 0.0, atol=1e-12)

    def test_direction_of_vertical_edge_is_horizontal(self):
        image = make_step(64)
        _, _, _, direction = sobel_gradients(image)
        # Gradient points along +x (intensity increases rightward),
        # so direction at the edge is ~0 radians.
        edge_dirs = direction[32, 31:33]
        np.testing.assert_allclose(edge_dirs, 0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()

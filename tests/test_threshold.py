"""Tests for fixed thresholding and Otsu's method."""

import unittest

import numpy as np

from edge_detect_lab.synthetic import make_square, make_step
from edge_detect_lab.threshold import fixed_threshold, otsu_binarize, otsu_threshold


class TestFixedThreshold(unittest.TestCase):
    def test_fixed_threshold(self):
        image = np.array([[0.1, 0.5], [0.49, 0.9]])
        mask = fixed_threshold(image, 0.5)
        np.testing.assert_array_equal(mask, [[False, True], [False, True]])


class TestOtsu(unittest.TestCase):
    def test_otsu_separates_bimodal_levels(self):
        # Bimodal image: half the pixels 0.2, half 0.8.
        image = make_step(64)
        thr = otsu_threshold(image)
        self.assertGreater(thr, 0.2)
        self.assertLess(thr, 0.8)

        mask, thr2 = otsu_binarize(image)
        self.assertEqual(thr, thr2)
        # The mask must recover exactly the bright half.
        expected = image >= 0.5
        np.testing.assert_array_equal(mask, expected)

    def test_otsu_on_square_recovers_foreground(self):
        image = make_square(64)
        mask, _ = otsu_binarize(image)
        np.testing.assert_array_equal(mask, image >= 0.5)

    def test_otsu_on_noisy_bimodal(self):
        rng = np.random.default_rng(42)
        dark = rng.normal(0.25, 0.05, size=(32, 32))
        bright = rng.normal(0.75, 0.05, size=(32, 32))
        image = np.hstack([dark, bright])
        thr = otsu_threshold(image)
        # Threshold should land between the two cluster means.
        self.assertGreater(thr, 0.3)
        self.assertLess(thr, 0.7)
        mask = fixed_threshold(image, thr)
        accuracy = (mask == (image >= 0.5)).mean()
        self.assertGreater(accuracy, 0.99)

    def test_otsu_constant_image(self):
        image = np.full((8, 8), 0.42)
        self.assertAlmostEqual(otsu_threshold(image), 0.42)


if __name__ == "__main__":
    unittest.main()

"""Tests for the convolution engine."""

import unittest

import numpy as np

from edge_detect_lab.convolution import convolve2d


class TestConvolution(unittest.TestCase):
    def test_identity_kernel_returns_input(self):
        image = np.arange(25, dtype=np.float64).reshape(5, 5) / 25.0
        identity = np.array([[0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0]])
        for padding in ("zero", "edge", "reflect"):
            out = convolve2d(image, identity, padding=padding)
            np.testing.assert_allclose(out, image)

    def test_known_kernel_known_input(self):
        # 3x3 image convolved (correlation form, flip=False) with a
        # box-average kernel. Expected values computed by hand with
        # zero padding.
        image = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
        box = np.ones((3, 3)) / 9.0
        out = convolve2d(image, box, padding="zero", flip=False)
        # Centre = mean of all 9 values = 45/9 = 5.
        self.assertAlmostEqual(out[1, 1], 5.0)
        # Corner (0,0): window covers [[0,0,0],[0,1,2],[0,4,5]] -> 12/9.
        self.assertAlmostEqual(out[0, 0], 12.0 / 9.0)
        # Edge (0,1): window [[0,0,0],[1,2,3],[4,5,6]] -> 21/9.
        self.assertAlmostEqual(out[0, 1], 21.0 / 9.0)

    def test_true_convolution_flips_kernel(self):
        # Kernel with a single 1 at top-left. True convolution flips
        # it to bottom-right, so the output is the image shifted by
        # (+1, +1): out[i, j] = image[i+1, j+1] (zero padded).
        image = np.arange(16, dtype=np.float64).reshape(4, 4)
        kernel = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]])
        out = convolve2d(image, kernel, padding="zero", flip=True)
        self.assertAlmostEqual(out[0, 0], image[1, 1])
        self.assertAlmostEqual(out[2, 2], image[3, 3])
        # Last row/col see only padding.
        self.assertAlmostEqual(out[3, 3], 0.0)

    def test_output_shape_preserved(self):
        image = np.random.default_rng(0).random((10, 7))
        kernel = np.ones((5, 3)) / 15.0
        out = convolve2d(image, kernel)
        self.assertEqual(out.shape, image.shape)

    def test_invalid_padding_raises(self):
        with self.assertRaises(ValueError):
            convolve2d(np.ones((3, 3)), np.ones((3, 3)), padding="bogus")


if __name__ == "__main__":
    unittest.main()

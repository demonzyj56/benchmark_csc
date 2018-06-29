#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for image dataset."""
import os
import unittest
import numpy as np
import image_dataset


class TestImageDataset(unittest.TestCase):
    """Test case for image_datset."""

    def test_image_items(self):
        """Check the validity of every image path"""
        for _, paths in image_dataset.get_image_items().items():
            if not isinstance(paths, list):
                paths = [paths]
            for path in paths:
                self.assertTrue(os.path.exists(path),
                                '{} does not exist'.format(path))

    def test_image_shape(self):
        """Check that loaded images have 3-D shape."""
        lena_path = image_dataset.get_image_items()['lena']
        img_color = image_dataset.load_image(lena_path, np.float32, gray=False)
        img_gray = image_dataset.load_image(lena_path, np.float32, gray=True)
        self.assertEqual(img_color.shape[-1], 3)
        self.assertEqual(img_gray.shape[-1], 1)

    def test_image_scale(self):
        """Check that scale flag works as expected."""
        lena_path = image_dataset.get_image_items()['lena']
        img = image_dataset.load_image(lena_path, np.float32, scaled=True)
        self.assertLessEqual(0., img.min())
        self.assertLessEqual(img.max(), 1.)
        img_unscaled = image_dataset.load_image(lena_path, np.float32,
                                                scaled=False)
        self.assertLessEqual(1., img_unscaled.max())

    def test_image_blob(self):
        """Check that returned image blob is 4-D."""
        fruits = image_dataset.create_image_blob('fruit', np.float32)
        self.assertSequenceEqual(fruits.shape, [100, 100, 3, 10])


if __name__ == "__main__":
    unittest.main()

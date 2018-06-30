#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for admm_slice."""
import unittest
import pyfftw  # pylint: disable=unused-import
import numpy as np
from sporco import util
from admm_slice import ConvBPDNSlice, ConvBPDNSliceTwoBlockCnstrnt
from test_fold import fold_naive, unfold_naive
import image_dataset


class TestConvBPDNSliceTwoBlockCnstrnt(unittest.TestCase):
    """Test Fixture for ConvBPDNSliceTwoBlockCnstrnt."""

    def setUp(self):
        """Initialize a ConvBPDNSliceTwoBlockCnstrnt object."""
        self.img = util.ExampleImages().image(
            'kodim23.png', scaled=True, gray=True, idxexp=np.s_[160:416, 60:316]
        )
        self.npd = 16
        self.fltlmbd = 10
        self.lmbda = 5e-2
        self.sl, self.sh = util.tikhonov_filter(self.img, self.fltlmbd, self.npd)
        self.D = util.convdicts()['G:12x12x36']
        opt = ConvBPDNSliceTwoBlockCnstrnt.Options({
            'Verbose': True, 'MaxMainIter': 200, 'RelStopTol': 5e-3,
            'Boundary': 'zeros_back'
        })
        self.solver = ConvBPDNSliceTwoBlockCnstrnt(self.D, self.sh,
                                                   lmbda=self.lmbda, opt=opt,
                                                   dimK=0)

    def test_shape_sanity(self):
        """Check the internal shape to be correct."""
        n, m = self.solver.D.shape
        self.assertEqual(n, 12*12)
        self.assertEqual(m, 36)
        self.assertSequenceEqual(self.solver.S.shape, [1, 1, 256, 256])
        self.assertSequenceEqual(self.solver.S_slice.shape, [1, 12*12, 256*256])
        self.assertSequenceEqual(self.solver.Y.shape, [1, n+m, 256*256])
        self.assertSequenceEqual(self.solver.U.shape, [1, n+m, 256*256])

class TestConvBPDNSlice(unittest.TestCase):
    """Test Fixture for ConvBPDNSlice."""

    def setUp(self):
        """Initialize a ConvBPDNSlice object."""
        self.img = util.ExampleImages().image(
            'kodim23.png', scaled=True, gray=True, idxexp=np.s_[160:416, 60:316]
        )
        self.npd = 16
        self.fltlmbd = 10
        self.lmbda = 5e-2
        self.sl, self.sh = util.tikhonov_filter(self.img, self.fltlmbd, self.npd)
        self.D = util.convdicts()['G:12x12x36']
        opt = ConvBPDNSlice.Options({'Verbose': True, 'MaxMainIter': 200,
                                     'RelStopTol': 5e-3,
                                     'Boundary': 'zeros_back'})
        self.solver = ConvBPDNSlice(self.D, self.sh, lmbda=self.lmbda, opt=opt,
                                    dimK=0)

    def test_shape_sanity(self):
        """The internal representation should have sane shape."""
        #  assert self.solver.D.shape == [12*12, 36]
        self.assertSequenceEqual(self.solver.D.shape, [12*12, 36])
        self.assertSequenceEqual(self.solver.S.shape,
                                 [1, 1, 256, 256])
        self.assertSequenceEqual(self.solver.S_slice.shape,
                                 [1, 12*12, 256*256])
        self.assertSequenceEqual(self.solver.Y.shape,
                                 [1, 12*12, 256*256])
        self.assertSequenceEqual(self.solver.U.shape,
                                 [1, 12*12, 256*256])
        self.solver.xstep()
        self.assertSequenceEqual(self.solver.X.shape,
                                 [1, 36, 256*256])

    def test_im2slices_zeros_back(self):
        """Check sanity of im2slices."""
        blob = np.random.randn(*self.solver.S.shape)
        out1 = self.solver.im2slices(blob)
        blob_padded = np.pad(blob, ((0, 0), (0, 0), (0, 11), (0, 11)), 'constant')
        out2 = unfold_naive(blob_padded, 12)
        self.assertTrue(np.allclose(out1, out2))

    def test_slices2im_zeros_back(self):
        """Check sanity of slices2im."""
        slices = np.random.randn(*self.solver.S_slice.shape)
        out1 = self.solver.slices2im(slices)
        out2 = fold_naive(slices, 256+12-1, 12)
        out2 = out2[:, :, :256, :256]
        self.assertTrue(np.allclose(out1, out2))


class TestConvBPDNSliceColor(unittest.TestCase):
    """Test fixture for ConvBPDNSlice for color image input."""

    def setUp(self):
        """Set image to be a color image."""
        self.img = image_dataset.create_image_blob('lena', gray=False)
        self.sl, self.sh = util.tikhonov_filter(self.img, 10, 16)
        self.D = util.convdicts()['RGB:8x8x3x64']
        self.lmbda = 0.05
        opt = ConvBPDNSlice.Options({
            'Verbose': True, 'MaxMainIter': 200,
            'RelStopTol': 5e-3, 'Boundary': 'zeros_back'
        })
        self.solver = ConvBPDNSlice(self.D, self.sh, lmbda=self.lmbda, opt=opt)

    def test_shape_sanity(self):
        """Test the internal shape is correct."""
        self.assertSequenceEqual(self.solver.D.shape, [8*8*3, 64])
        self.assertSequenceEqual(self.solver.S_slice.shape,
                                 [1, 8*8*3, 512*512])
        self.assertSequenceEqual(self.solver.Y.shape,
                                 [1, 8*8*3, 512*512])
        self.assertSequenceEqual(self.solver.U.shape,
                                 [1, 8*8*3, 512*512])
        self.solver.xstep()
        self.assertSequenceEqual(self.solver.X.shape,
                                 [1, 64, 512*512])


if __name__ == "__main__":
    unittest.main()

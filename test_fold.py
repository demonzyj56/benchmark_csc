"""Test script for fold and unfold."""
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from fold_naive import fold_naive, unfold_naive


class TestFoldAndUnfold(unittest.TestCase):

    def setUp(self):
        """Setup test blob."""
        self.blob = np.random.randn(100, 3, 32, 32)
        self.rslice = np.random.randn(100, 3*11*11, 22*22)
        self.kernel_size = 11
        self.output_size = 32

    def test_unfold_equal(self):
        unfolded_numpy = unfold_naive(self.blob, self.kernel_size)
        with torch.no_grad():
            blob_torch = torch.from_numpy(self.blob)
            unfolded_torch = F.unfold(blob_torch, self.kernel_size)
        self.assertTrue(np.allclose(unfolded_numpy, unfolded_torch.numpy()))

    def _run_fold_equal(self, unfolded):
        recon_numpy = fold_naive(unfolded, self.blob.shape[2], self.kernel_size)
        with torch.no_grad():
            unfolded_torch = torch.from_numpy(unfolded)
            recon_torch = F.fold(unfolded_torch, self.blob.shape[2], self.kernel_size)
        self.assertTrue(np.allclose(recon_numpy, recon_torch.numpy()))

    def test_unfold_fold_equal(self):
        unfolded = unfold_naive(self.blob, self.kernel_size)
        self._run_fold_equal(unfolded)

    def test_random_fold_equal(self):
        self._run_fold_equal(self.rslice)


if __name__ == "__main__":
    unittest.main()

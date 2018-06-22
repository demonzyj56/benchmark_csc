"""Naive implementation of fold and unfold."""
import numpy as np


def unfold_naive(blob, kernel_h, kernel_w=None):
    """Naive impl."""
    if kernel_w is None:
        kernel_w = kernel_h
    in_channels = blob.shape[1] * kernel_h * kernel_w
    H, W = blob.shape[2]-kernel_h+1, blob.shape[3]-kernel_w+1
    num_slices_per_batch = H * W
    slices = np.zeros(
        (blob.shape[0], in_channels, num_slices_per_batch),
        dtype=blob.dtype
    )
    for n in range(blob.shape[0]):
        for h in range(H):
            for w in range(W):
                s = blob[n, :, h:h+kernel_h, w:w+kernel_w].ravel()
                slices[n, :, h*W+w] = s
    return slices


def fold_naive(slices, output_size, kernel_size):
    """Naive impl."""
    channels = slices.shape[1] // kernel_size // kernel_size
    blob = np.zeros(
        (slices.shape[0], channels, output_size, output_size),
        dtype=slices.dtype
    )
    H, W = output_size-kernel_size+1, output_size-kernel_size+1
    for n in range(blob.shape[0]):
        for h in range(H):
            for w in range(W):
                s = slices[n, :, h*W+w]
                blob[n, :, h:h+kernel_size, w:w+kernel_size] += \
                    s.reshape(channels, kernel_size, kernel_size)
    return blob

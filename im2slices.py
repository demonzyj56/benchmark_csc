"""Utility functions for converting 4-D blobs to slices and vice versa."""
import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _pad_circulant_front(blob, pad_h, pad_w):
    """Pad a 4-D blob with circulant boundary condition at the front."""
    return np.pad(blob, ((0, 0), (0, 0), (pad_h, 0), (pad_w, 0)), 'wrap')


def _pad_circulant_back(blob, pad_h, pad_w):
    """Pad a 4-D blob with circulant boundary condition at the back."""
    return np.pad(blob, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 'wrap')


def _pad_zeros_front(blob, pad_h, pad_w):
    """Pad a 4-D blob with zero boundary condition at the front."""
    return np.pad(blob, ((0, 0), (0, 0), (pad_h, 0), (pad_w, 0)), 'constant')


def _pad_zeros_back(blob, pad_h, pad_w):
    """Pad a 4-D blob with zero boundary condition at the back."""
    return np.pad(blob, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 'constant')


def _crop_circulant_front(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for circulant boundary condition
    at the front."""
    cropped = blob[:, :, pad_h:, pad_w:]
    cropped[:, :, -pad_h:, :] += blob[:, :, :pad_h, pad_w:]
    cropped[:, :, :, -pad_w:] += blob[:, :, pad_h:, :pad_w]
    cropped[:, :, -pad_h:, -pad_w:] += blob[:, :, :pad_h, :pad_w]
    return cropped


def _crop_circulant_back(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for circulant boundary condition
    at the back."""
    cropped = blob[:, :, :-pad_h, :-pad_w]
    cropped[:, :, :pad_h, :] += blob[:, :, -pad_h:, :-pad_w]
    cropped[:, :, :, :pad_w] += blob[:, :, :-pad_h, -pad_w:]
    cropped[:, :, :pad_h, :pad_w] += blob[:, :, -pad_h:, -pad_w:]
    return cropped


def _crop_zeros_front(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for zero boundary condition
    at the front."""
    return blob[:, :, pad_h:, pad_w:]


def _crop_zeros_back(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for zero boundary condition
    at the back."""
    return blob[:, :, :-pad_h, :-pad_w]


def im2slices(S, kernel_h, kernel_w, boundary='circulant_back'):
    r"""Convert the input signal :math:`S` to a slice form.
    Assuming the input signal having a standard shape as pytorch variable
    (N, C, H, W).  The output slices have shape
    (batch_size, slice_dim, num_slices_per_batch).
    """
    pad_h, pad_w = kernel_h - 1, kernel_w - 1
    S_torch = globals()['_pad_{}'.format(boundary)](S, pad_h, pad_w)
    with torch.no_grad():
        S_torch = torch.from_numpy(S_torch)  # pylint: disable=no-member
        slices = F.unfold(S_torch, kernel_size=(kernel_h, kernel_w))
    return slices.numpy()


def slices2im(slices, kernel_h, kernel_w, output_h, output_w,
              boundary='circulant_back'):
    r"""Reconstruct input signal :math:`\hat{S}` for slices.
    The input slices should have compatible size of
    (batch_size, slice_dim, num_slices_per_batch), and the
    returned signal has shape (N, C, H, W) as standard pytorch variable.
    """
    pad_h, pad_w = kernel_h - 1, kernel_w - 1
    with torch.no_grad():
        slices_torch = torch.from_numpy(slices)  # pylint: disable=no-member
        S_recon = F.fold(
            slices_torch, (output_h+pad_h, output_w+pad_w), (kernel_h, kernel_w)
        )
    S_recon = globals()['_crop_{}'.format(boundary)](
        S_recon.numpy(), pad_h, pad_w
    )
    return S_recon

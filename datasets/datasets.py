"""Utility function for loading different dataset."""
import os.path as osp
import numpy as np
import scipy.io as sio
from .cifar import CIFAR10
from .image_dataset import create_image_blob
from .flower import VGG17Flowers, VGG102Flowers


__ROOT = osp.join(osp.dirname(__file__), '..')


def get_dataset(name, root=None, train=True, dtype=np.float32, scaled=True,
                gray=False):
    """Get the dataset by name"""
    if name == 'fruit' or name == 'city':
        if gray:
            tt = 'train' if train else 'test'
            if root is None:
                root = osp.join(__ROOT, 'images', 'OCSC')
            path = osp.join(root, name+'_10', tt, '%s_lcne.mat' % tt)
            assert osp.exists(path)
            blob = sio.loadmat(path)['b'].astype(dtype)
        else:
            if not train:
                name = 'singles'
            blob = create_image_blob(name, dtype, scaled=scaled, gray=False)
        return blob
    elif name == 'cifar10':
        if root is None:
            root = osp.join(__ROOT, '.cifar10')
        cifar10 = CIFAR10(root=root, train=train, download=True,
                          data_type=dtype)
        if train:
            blob = cifar10.train_data
        else:
            blob = cifar10.test_data
        if scaled:
            blob /= 255.
        blob = blob.transpose(2, 3, 1, 0)
        if gray:
            blob = blob[:, :, 0, :] * 0.2989 + blob[:, :, 1, :] * 0.5870 + \
                blob[:, :, 2, :] * 0.1140
        del cifar10
        return blob
    elif name == '17flowers':
        if root is None:
            root = osp.join(__ROOT, '.17flowers')
        dataset = VGG17Flowers(root=root, train=train, dtype=dtype,
                               scaled=scaled, gray=gray)
        return dataset
    elif name == '102flowers':
        if root is None:
            root = osp.join(__ROOT, '.102flowers')
        dataset = VGG102Flowers(root=root, train=train, dtype=dtype,
                                scaled=scaled, gray=gray)
        return dataset
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

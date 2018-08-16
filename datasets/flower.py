"""Loader for VGG Flower 17/102 dataset."""
import os
import numpy as np
import scipy.io as sio
import torch.utils.data as data
import cv2


class VGG17Flowers(data.Dataset):
    """VGG 17Flowers dataset."""

    def __init__(self, root, train=True, dtype=np.float32, scaled=True,
                 gray=False, dsize=None):
        with open(os.path.join(root, 'files.txt'), 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            allfiles = [os.path.join(root, l) for l in lines]
        assert all([os.path.exists(f) for f in allfiles])
        mat = sio.loadmat(os.path.join(root, 'datasplits.mat'))
        if train:
            fileidx = mat['trn1'].ravel().tolist() + mat['val1'].ravel().tolist()
        else:
            fileidx = mat['tst1'].ravel().tolist()
        self.files = [allfiles[idx-1] for idx in fileidx]
        self.train = train
        self.dtype = dtype
        self.scaled = scaled
        self.gray = gray
        self.dsize = dsize if dsize is not None else (256, 256)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Notice that by default we rescale each image to 500x500.
        path = self.files[index]
        if self.gray:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=self.dsize)
        image = image.astype(self.dtype)
        if self.scaled and not np.issubdtype(image.dtype, np.integer):
            image /= 255.
        return image


class VGG102Flowers(data.Dataset):
    """VGG 102Flowers dataset."""

    def __init__(self, root, train=True, dtype=np.float32, scaled=True,
                 gray=False, dsize=None):
        allfiles = filter(lambda s: s.endswith('jpg'), os.listdir(root))
        allfiles = [os.path.join(root, f) for f in allfiles]
        assert all([os.path.exists(f) for f in allfiles])
        mat = sio.loadmat(os.path.join(root, 'setid.mat'))
        if train:
            fileidx = mat['trnid'].ravel().tolist() + mat['valid'].ravel().tolist()
        else:
            fileidx = mat['tstid'].ravel().tolist()
        self.files = [allfiles[idx-1] for idx in fileidx]
        self.train = train
        self.dtype = dtype
        self.scaled = scaled
        self.gray = gray
        self.dsize = dsize if dsize is not None else (256, 256)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        if self.gray:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=self.dsize)
        image = image.astype(self.dtype)
        if self.scaled and not np.issubdtype(image.dtype, np.integer):
            image /= 255.
        return image

"""Loader for VOC2007 dataset, with images only."""
import os
import numpy as np
import torch.utils.data as data
import cv2


class VOC07Images(data.Dataset):
    """Pascal VOC 2007 images"""
    def __init__(self, root, train=True, dtype=np.float32, scaled=True,
                 gray=False, dsize=None):
        if train:
            img_list = os.path.join(root, 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')
        else:
            img_list = os.path.join(root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
        assert os.path.exists(img_list), 'Path does not exists: {}'.format(img_list)
        with open(img_list, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        img_path_template = os.path.join(root, 'VOC2007', 'JPEGImages', '%s.jpg')
        self.img_path = [img_path_template % l for l in lines]
        assert all([os.path.exists(p) for p in self.img_path])
        self.train = train
        self.dtype = dtype
        self.scaled = scaled
        self.gray = gray
        self.dsize = dsize if dsize is not None else (256, 256)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.img_path[index]
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

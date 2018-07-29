"""Reimplementation of torchvision.datasets.cifar because we are not using
PIL images."""
import numpy as np
from torch.utils.data import Dataset
import torchvision.datasets.cifar as tcifar


class CIFAR10(Dataset):

    cifar_func = tcifar.CIFAR10

    def __init__(self, root, train=True, download=False, data_type=np.float64):
        """Image blob are organized as NCHW."""
        cifar_data = self.cifar_func(root, train=train, download=download)
        self.train = train
        if self.train:
            self.train_data = cifar_data.train_data.transpose((0, 3, 1, 2))
            self.train_data = self.train_data.astype(data_type)
            self.train_labels = np.array(cifar_data.train_labels, dtype=np.int32)

        else:
            self.test_data = cifar_data.test_data.transpose((0, 3, 1, 2))
            self.test_data = self.test_data.astype(data_type)
            self.test_labels = np.array(cifar_data.test_labels, dtype=np.int32)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)


class CIFAR100(CIFAR10):

    cifar_func = tcifar.CIFAR100

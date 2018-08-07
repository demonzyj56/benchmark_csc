"""Abstraction of loader for different cases."""
import numpy as np
import torch.utils.data as data


class BlobLoader(object):
    """An iterator to wrap over data."""

    def __init__(self, dataset, epochs=None, batch_size=None):
        self.dataset = dataset
        if isinstance(self.dataset, data.Dataset):
            self.size = len(self.dataset)
        else:
            self.size = self.dataset.shape[-1]
        if epochs is not None:
            self.epochs = epochs
        else:
            self.epochs = self.size
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        assert self.batch_size <= self.size
        self.e = 0
        self.cur = 0
        self.idx = np.random.permutation(self.size)

    def __iter__(self):
        self.e = 0
        self.cur = 0
        self.idx = np.random.permutation(self.size)
        return self

    def __next__(self):
        if self.e < self.epochs:
            if self.cur + self.batch_size > len(self.idx):
                self.cur = 0
                self.idx = np.random.permutation(self.size)
            selected = self._get_next_minibatch()
            self.cur += self.batch_size
            self.e += 1
            return selected
        else:
            raise StopIteration

    def _get_next_minibatch(self):
        """Return the desired minibatch."""
        if isinstance(self.dataset, data.Dataset):
            imgs = [self.dataset[self.idx[i]] for i in
                    range(self.cur, self.cur+self.batch_size)]
            imgs = np.stack(imgs, -1)
        else:
            imgs = self.dataset[...,
                                self.idx[self.cur:self.cur+self.batch_size]]
        return imgs

    def random_sample(self):
        """Since all solvers require a random sample to initialize, here a
        random sample is provided with correct shape."""
        if isinstance(self.dataset, data.Dataset):
            sample = self.dataset[0]
            return sample[..., None]  # add a batch idx
        return self.dataset[..., [0]]

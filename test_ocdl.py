#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for online convolutional sparse coding."""
import argparse
import datetime
import glob
import logging
import pickle
import os
import sys
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import sporco.util as su
import sporco.metric as sm
from sporco.admm import cbpdn
import image_dataset
from cifar import CIFAR10


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Options for benchmarking dictlrn algorithms CDL.')
    parser.add_argument('--name', default='ocdl', type=str, help='Name for experiment')
    parser.add_argument('--output_path', default='.default_ocdl', type=str, help='Path for output')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
    parser.add_argument('--lambda', '--lmbda', dest='lmbda', default=0.2, type=float, help='Lambda value for CDL')
    parser.add_argument('--cfg', default='.default_ocdl.yml', type=str, help='Config yaml file for solvers')
    parser.add_argument('--rng_seed', default=-1, type=int, help='Random seed to use; negative values mean dont set')
    parser.add_argument('--no_tikhonov_filter', action='store_true', help='No tikhonov low pass filtering is applied')
    parser.add_argument('--num_samples', default=-1, type=int, help='Number of test samples to use; -1 means use all')
    return parser.parse_args()


def setup_logging(name, filename=None):
    """Utility for every script to call on top-level.
    If filename is not None, then also log to the filename."""
    FORMAT = '[%(levelname)s %(asctime)s] %(filename)s:%(lineno)4d: %(message)s'
    DATEFMT = '%Y-%m-%d %H:%M:%S'
    logging.root.handlers = []
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename, mode='w'))
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt=DATEFMT,
        handlers=handlers
    )
    return logging.getLogger(name)


def get_stats_from_dict(Dd, args, test_blob):
    """Test function for obtaining functional values and PSNR for each
    data point at test set.

    Parameters
    ----------
    Dd: dict
        A dict mapping class names to a list of convolutional dictionaries,
        each of which is a snapshot at one timestep.
    args: ArgumentParser
        Arguments from argument parser.
    test_blob: numpy.ndarray
        If not None, then it holds the test images.
    """
    logger = logging.getLogger(__name__)
    opt = yaml.load(open(args.cfg, 'r'))
    opt = cbpdn.ConvBPDN.Options(opt.get('CBPDN', None))
    if not args.no_tikhonov_filter:
        sl, sh = su.tikhonov_filter(test_blob, 5.)
    else:
        sl = 0.
        sh = test_blob
    model_stats = {}
    for name, Ds in Dd.items():
        logger.info('Testing solver %s', name)
        stats = []
        for idx, D in enumerate(Ds):
            solver = cbpdn.ConvBPDN(D, sh, args.lmbda, opt=opt)
            solver.solve()
            fnc = solver.getitstat().ObjFun[-1]
            shr = solver.reconstruct().squeeze()
            imgr = sl + shr
            psnr = sm.psnr(test_blob, imgr)
            stats.append((fnc, psnr))
            del solver, imgr
            logger.info('%s %d/%d: ObjFun: %.2f, PSNR: %.2f',
                        name, idx+1, len(Ds), fnc, psnr)
        model_stats.update({name: stats})
    path = os.path.join(args.output_path, 'final_stats.pkl')
    logger.info('Saving final statistics to %s', path)
    pickle.dump(model_stats, open(path, 'wb'))
    return model_stats


def load_stats_from_folder(args):
    """Load dictionaries and other statistics from output path."""
    assert os.path.exists(args.output_path)
    cfg = yaml.load(open(args.cfg, 'r'))
    del cfg['CBPDN']
    Dd = {}
    stats = {}
    for k in cfg.keys():
        # load all dicts
        p = os.path.join(args.output_path, k)
        assert os.path.exists(p)
        npy_path = os.path.join(p, '{:s}.*.npy'.format(args.dataset))
        npy_path = glob.glob(npy_path)
        assert all([os.path.exists(pp) for pp in npy_path])
        # sort paths according to index
        npy_path = sorted(npy_path, key=lambda t: int(t.split('.')[-2]))
        Ds = [np.load(pp) for pp in npy_path]
        Dd.update({k: Ds})

        # load statistics
        stats_p = os.path.join(args.output_path, '{:s}.{:s}_stats.npy'.format(
            args.dataset, k
        ))
        assert os.path.exists(stats_p)
        stats.update({k: np.load(stats_p)})

    return Dd, stats


def plot_statistics(args, time_stats, fnc_stats=None):
    """Plot obtain statistics."""
    raise NotImplementedError


def dataset_loader(name, args):
    """Return train and test dataset by name."""
    if name == 'fruit' or name == 'city':
        train_blob = image_dataset.create_image_blob(name, np.float32,
                                                     scaled=True, gray=False)
        test_blob = image_dataset.create_image_blob('singles', np.float32,
                                                    scaled=True, gray=False)
    elif name == 'cifar10':
        cifar10_train = CIFAR10(root='.cifar10', train=True, download=True,
                                data_type=np.float32)
        cifar10_test = CIFAR10(root='.cifar10', train=False, download=True,
                               data_type=np.float32)
        train_blob = cifar10_train.train_data / 255.
        test_blob = cifar10_test.test_data / 255.
        train_blob = train_blob.transpose(2, 3, 1, 0)
        test_blob = test_blob.transpose(2, 3, 1, 0)
        del cifar10_train, cifar10_test
    else:
        raise NotImplementedError
    return (train_blob, test_blob)


def main():
    """Main entry."""
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    log_name = os.path.join(
        args.output_path,
        '{:s}.{:s}.{:%Y-%m-%d_%H-%M-%S}.log'.format(
            args.name,
            args.dataset,
            datetime.datetime.now(),
        )
    )
    logger = setup_logging(__name__, log_name)

    # set random seed
    if args.rng_seed >= 0:
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)

    _, test_blob = dataset_loader(args.dataset, args)
    if args.num_samples > 0 and args.num_samples < test_blob.shape[-1]:
        logger.info('Sampling test images: %d -> %d', test_blob.shape[-1],
                    args.num_samples)
        selected = np.random.choice(test_blob.shape[-1], args.num_samples,
                                    replace=False)
        test_blob = test_blob[..., selected]
    Dd, time_stats = load_stats_from_folder(args)
    fnc_stats = get_stats_from_dict(Dd, args, test_blob)
    time_stats = {k: v[0][v[1].index('Time')] for k, v in time_stats.items()}
    plot_statistics(args, time_stats, fnc_stats)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark code for online convolutional sparse coding algorithms.

Dataset: fruit/house/cifar10/flower.

Algorithms: TBD

"""
import argparse
import datetime
import logging
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
from sporco.dictlrn.onlinecdl import OnlineConvBPDNDictLearn
from fista_slice_online import OnlineDictLearnSliceSurrogate
from onlinecdl_surrogate import OnlineDictLearnDenseSurrogate
import image_dataset
from cifar import CIFAR10


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Options for benchmarking dictlrn algorithms CDL.')
    parser.add_argument('--name', default='ocdl', type=str, help='Name for experiment')
    parser.add_argument('--output_path', default='.default_ocdl', type=str, help='Path for output')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
    parser.add_argument('--lambda', '--lmbda', dest='lmbda', default=0.2, type=float, help='Lambda value for CDL')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of samples for each epoch')
    parser.add_argument('--cfg', default='.default_ocdl.yml', type=str, help='Config yaml file for solvers')
    parser.add_argument('--patch_size', default=8, type=int, help='Height and width for dictionary')
    parser.add_argument('--num_atoms', default=100, type=int, help='Size of dictionary')
    parser.add_argument('--rng_seed', default=-1, type=int, help='Random seed to use; negative values mean dont set')
    parser.add_argument('--no_tikhonov_filter', action='store_true', help='No tikhonov low pass filtering is applied')
    parser.add_argument('--skip_test', action='store_true', help='Skip running reconstruction')
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


def train_models(solvers, train_blob, args):
    """Function for training every solvers."""
    cur = 0
    idx = np.random.permutation(train_blob.shape[-1])
    # setup for every solver
    paths = {}
    for name in solvers.keys():
        path = os.path.join(args.output_path, name)
        if not os.path.exists(path):
            os.makedirs(path)
        paths.update({name: path})
    for e in range(args.epochs):
        if cur + args.batch_size > len(idx):
            cur = 0
            idx = np.random.permutation(train_blob.shape[-1])
        # step all solvers to ensure that for each step they receive
        # same data
        for k, s in solvers.items():
            blob = train_blob[..., idx[cur:cur+args.batch_size]]
            if not args.no_tikhonov_filter:
                # fix lambda to be 5
                _, blob = su.tikhonov_filter(blob, 5.)
            s.solve(blob)
            np.save(os.path.join(paths[k], f'{args.dataset}.{e}.npy'),
                    s.getdict().squeeze())
        cur += args.batch_size
    return solvers


def reconstruct_psnr(solvers, test_blob, args, opt=None):
    """Compute reonstruction PSNR on test samples."""
    logger = logging.getLogger(__name__)
    if not args.no_tikhonov_filter:
        sl, sh = su.tikhonov_filter(test_blob, 5.)
    else:
        sl = 0.
        sh = test_blob
    psnr_all = {}
    for k, v in solvers.items():
        d = cbpdn.ConvBPDN(v.getdict().squeeze(), sh, args.lmbda, opt=opt)
        d.solve()
        shr = d.reconstruct().squeeze()
        imgr = sl + shr
        psnr = sm.psnr(test_blob, imgr)
        logger.info('Reconstruction PSNR for {:s}: {:.3f}dB'.format(k, psnr))
        psnr_all.update({k: psnr})
    return psnr_all


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


def plot_and_save_statistics(solvers, args):
    """Plot some desired statistics."""
    for k, v in solvers.items():
        # save dictionaries visualization
        plt.clf()
        plt.imshow(su.tiledict(v.getdict().squeeze()))
        plt.savefig(os.path.join(args.output_path, f'{args.dataset}.{k}.pdf'),
                    bbox_inches='tight')
        # save statistics
        stats_arr = su.ntpl2array(v.getitstat())
        np.save(os.path.join(args.output_path, f'{args.dataset}.{k}_stats.npy'),
                stats_arr)
    if 1:
        plt.clf()
        nsol = len(solvers)
        for i, (k, v) in enumerate(solvers.items()):
            plt.subplot(1, nsol, i+1)
            plt.imshow(su.tiledict(v.getdict().squeeze()))
            plt.title(k)
        plt.show()


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

    # load configs
    cfg = yaml.load(open(args.cfg, 'r'))
    copt = cbpdn.ConvBPDN.Options(cfg['ConvBPDN'])
    del cfg['ConvBPDN']

    # load data
    train_blob, test_blob = dataset_loader(args.dataset, args)

    # load dictionary
    # TODO(leoyolo): check for grayscale image
    D0 = np.random.randn(args.patch_size, args.patch_size, 3, args.num_atoms)

    # load solvers
    solvers = {}
    for k, v in cfg.items():
        solver_class = globals()[k.split('-')[0]]
        opt = solver_class.Options(v)
        solvers.update({
            k: solver_class(D0, train_blob[..., [0]], args.lmbda, opt=opt)
        })

    # train solvers
    solvers = train_models(solvers, train_blob, args)

    # plot and save everything
    plot_and_save_statistics(solvers, args)

if __name__ == "__main__":
    main()

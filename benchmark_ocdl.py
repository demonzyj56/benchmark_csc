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
import pprint
import pickle
import sys
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import sporco.util as su
import sporco.metric as sm
from sporco.dictlrn.onlinecdl import OnlineConvBPDNDictLearn
from fista_slice_online import OnlineDictLearnSliceSurrogate
from onlinecdl_surrogate import OnlineDictLearnDenseSurrogate
from datasets import get_dataset, BlobLoader
from utils import setup_logging


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
    parser.add_argument('--use_gray', action='store_true', help='Use grayscale image instead of color image')
    return parser.parse_args()


def train_models(solvers, train_blob, args):
    """Function for training every solvers."""
    if args.use_gray:
        D0 = np.random.randn(args.patch_size, args.patch_size, args.num_atoms)
    else:
        D0 = np.random.randn(args.patch_size, args.patch_size, 3, args.num_atoms)

    loader = BlobLoader(train_blob, args.epochs, args.batch_size)
    sample = loader.random_sample()
    solvers = {k: sol_name(D0, sample, args.lmbda, opt=opt) for
               k, (sol_name, opt) in solvers.items()}
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    for e, blob in enumerate(loader):
        if not args.no_tikhonov_filter:
            # fix lambda to be 5
            _, blob = su.tikhonov_filter(blob, 5.)
        for k, s in solvers.items():
            s.solve(blob)
            path = os.path.join(args.output_path, k)
            np.save(os.path.join(path, '{}.{}.npy'.format(dname, e)),
                    s.getdict().squeeze())
    # snapshot iteration record
    sio.savemat(os.path.join(args.output_path, 'iter_record.mat'),
                {'iter_record': loader.record})
    return solvers


def plot_and_save_statistics(solvers, args):
    """Plot some desired statistics."""
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    for k, v in solvers.items():
        # save dictionaries visualization
        plt.clf()
        d = v.getdict().squeeze()
        if d.ndim == 3:  # grayscale image
            plt.imshow(su.tiledict(d), cmap='gray')
        else:
            plt.imshow(su.tiledict(d))
        plt.savefig(os.path.join(args.output_path, k, f'{dname}.pdf'),
                    bbox_inches='tight')
        # save statistics
        stats_arr = su.ntpl2array(v.getitstat())
        np.save(os.path.join(args.output_path, k, f'{dname}.stats.npy'),
                stats_arr)
        # we save time separately
        time_stats = {'Time': v.getitstat().Time}
        pickle.dump(
            time_stats,
            open(os.path.join(args.output_path, k,
                              f'{dname}.time_stats.pkl'), 'wb')
        )
    if 1:
        plt.clf()
        nsol = len(solvers)
        for i, (k, v) in enumerate(solvers.items()):
            plt.subplot(1, nsol, i+1)
            d = v.getdict().squeeze()
            if d.ndim == 3:  # grayscale image
                plt.imshow(su.tiledict(d), cmap='gray')
            else:
                plt.imshow(su.tiledict(d))
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
    logger.info('args:')
    logger.info(pprint.pformat(args))

    # set random seed
    if args.rng_seed >= 0:
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)

    # load configs
    cfg = yaml.load(open(args.cfg, 'r'))
    del cfg['ConvBPDN']
    logger.info('cfg:')
    logger.info(pprint.pformat(cfg))

    # load data
    train_blob = get_dataset(args.dataset, gray=args.use_gray, train=True)

    # load solvers
    solvers = {}
    for k, v in cfg.items():
        solver_class = globals()[k.split('-')[0]]
        opt = solver_class.Options(v)
        path = os.path.join(args.output_path, k)
        if not os.path.exists(path):
            os.makedirs(path)
        solvers.update({k: (solver_class, opt)})

    # train solvers
    solvers = train_models(solvers, train_blob, args)

    # plot and save everything
    plot_and_save_statistics(solvers, args)


if __name__ == "__main__":
    main()

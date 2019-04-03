#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script for running test on fruit/gray dataset."""
import argparse
import datetime
import logging
import pickle
import pprint
import re
import os
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
from sporco.admm import cbpdn
from sporco.admm import parcbpdn
from utils import setup_logging
from datasets import get_dataset
from test_ocdl import load_dict, get_stats_from_dict, load_time_stats, \
    plot_statistics, get_stats_from_dict_gpu


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Options for benchmarking dictlrn algorithms CDL.')
    parser.add_argument('--name', default='ocdl', type=str, help='Name for experiment')
    parser.add_argument('--output_path', default='.fruit.gray.noise_free', type=str, help='Path for output')
    parser.add_argument('--dataset', default='fruit', type=str, help='Dataset to use')
    parser.add_argument('--lambda', '--lmbda', dest='lmbda', default=0.2, type=float, help='Lambda value for CDL')
    parser.add_argument('--cfg', default='experiments/fruit.gray.yml', type=str, help='Config yaml file for solvers')
    parser.add_argument('--rng_seed', default=-1, type=int, help='Random seed to use; negative values mean dont set')
    parser.add_argument('--no_tikhonov_filter', default=True, type=bool, help='No tikhonov low pass filtering is applied')
    parser.add_argument('--use_gray', default=True, type=bool, help='Use grayscale image instead of color image')
    parser.add_argument('--last_only', action='store_true', help='Only compute the statistics from last epoch')
    parser.add_argument('--use_gpu', action='store_true', help='Whether to use gpu impl of CBPDN to compute')
    parser.add_argument('--pad_boundary', action='store_true', help='Pad the boundary of test_blob with zeros')
    parser.add_argument('--no_legacy', action='store_true', help='Remove some legacy code for config')
    return parser.parse_args()


def load_OCSC_dict(args):
    """Load dictionaries trained by OCSC."""
    path = os.path.join(args.output_path, 'OCSC')
    assert os.path.exists(path)
    files = os.listdir(path)
    pattern = re.compile(r'fruit\.gray\.[0-9]+\.mat')
    files = [f for f in files if pattern.match(f) is not None]
    mat_path = [os.path.join(path, f) for f in files]
    assert all([os.path.exists(pp) for pp in mat_path])
    # sort paths according to index
    mat_path = sorted(mat_path, key=lambda t: int(t.split('.')[-2]))
    Ds = [sio.loadmat(pp)['d'].astype(np.float32) for pp in mat_path]
    return {'OCSC': Ds}


def load_OCSC_time_stats(args):
    """Load time statistics from OCSC."""
    path = os.path.join(args.output_path, 'OCSC', 'fruit.gray.time_stats.mat')
    assert os.path.exists(path)
    time_stats = sio.loadmat(path)['tt'].ravel().cumsum().tolist()
    return {'OCSC': time_stats}


def main():
    """Main entry."""
    args = parse_args()
    assert os.path.exists(args.output_path)
    if (not args.no_legacy) and \
            (not os.path.exists(os.path.join(args.output_path, 'OCSC'))):
        os.system('ln -s /backup/OCSC/result/fruit_10/ {:s}'.format(
            os.path.join(args.output_path, 'OCSC')
        ))
    log_name = os.path.join(
        args.output_path,
        '{:s}.{:s}.{:%Y-%m-%d_%H-%M-%S}.test.log'.format(
            args.name, 'fruit.gray', datetime.datetime.now(),
        )
    )
    logger = setup_logging(__name__, log_name)
    logger.info('args')
    logger.info(pprint.pformat(args))

    # set random seed
    if args.rng_seed >= 0:
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)

    test_blob = get_dataset('fruit', train=False, gray=True)

    # load all snapshots of dictionaries
    Dd = load_dict(args)
    Dd_ocsc = load_OCSC_dict(args)
    Dd.update(Dd_ocsc)

    # load time statistics
    time_stats = load_time_stats(args)
    time_stats_ocsc = load_OCSC_time_stats(args)
    time_stats.update(time_stats_ocsc)

    if not args.last_only:

        # compute functional values and PSNRs from dictionaries
        if not args.use_gpu:
            fnc_stats = get_stats_from_dict(Dd, args, test_blob)
        else:
            fnc_stats = get_stats_from_dict_gpu(Dd, args, test_blob)

        # plot statistics
        plot_statistics(args, time_stats, fnc_stats)

    else:
        Dd = {k: [v[-1]] for k, v, in Dd.items()}
        if not args.use_gpu:
            get_stats_from_dict(Dd, args, test_blob)
        else:
            get_stats_from_dict_gpu(Dd, args, test_blob)


if __name__ == "__main__":
    main()

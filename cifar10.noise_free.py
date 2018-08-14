#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for CIFAR10."""
import argparse
import datetime
import pprint
import os
import sys
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from utils import setup_logging
from datasets import get_dataset
from test_ocdl import load_dict, get_stats_from_dict, load_time_stats, \
    plot_statistics


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Options for benchmarking dictlrn algorithms CDL.')
    parser.add_argument('--name', default='ocdl', type=str, help='Name for experiment')
    parser.add_argument('--output_path', default='.cifar10.noise_free', type=str, help='Path for output')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset to use')
    parser.add_argument('--lambda', '--lmbda', dest='lmbda', default=0.2, type=float, help='Lambda value for CDL')
    parser.add_argument('--cfg', default='experiments/cifar10.yml', type=str, help='Config yaml file for solvers')
    parser.add_argument('--rng_seed', default=-1, type=int, help='Random seed to use; negative values mean dont set')
    parser.add_argument('--no_tikhonov_filter', default=False, type=bool, help='No tikhonov low pass filtering is applied')
    parser.add_argument('--use_gray', default=False, type=bool, help='Use grayscale image instead of color image')
    parser.add_argument('--last_only', action='store_true', help='Only compute the statistics from last epoch')
    parser.add_argument('--pad_boundary', action='store_true', help='Pad the boundary of test_blob with zeros')
    parser.add_argument('--num_test', default=-1, type=int, help='Number of test samples; negative values mean use all')
    return parser.parse_args()


def main():
    """Main entry."""
    args = parse_args()
    assert os.path.exists(args.output_path)
    log_name = os.path.join(
        args.output_path,
        '{:s}.{:s}.{:%Y-%m-%d_%H-%M-%S}.test.log'.format(
            args.name,
            args.dataset,
            datetime.datetime.now(),
        )
    )
    logger = setup_logging(__name__, log_name)
    logger.info('args')
    logger.info(pprint.pformat(args))

    # set random seed
    if args.rng_seed >= 0:
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)

    test_blob = get_dataset(args.dataset, train=False, gray=False)
    # sample test set
    if args.num_test > 0 and args.num_test < test_blob.shape[-1]:
        index_name = os.path.join(args.output_path,
                                  f'{args.dataset}.test_idx.{args.num_test}.npy')
        if os.path.exists(index_name):
            selected = np.load(index_name)
        else:
            selected = np.random.choice(test_blob.shape[-1], args.num_test,
                                        replace=False)
            np.save(index_name, selected)
        logger.info('Selecting %d -> %d test samples from %s.',
                    test_blob.shape[-1], args.num_test, args.dataset)
        test_blob = test_blob[..., selected]

    Dd = load_dict(args)

    fnc_stats = get_stats_from_dict(Dd, args, test_blob)

    # load time statistics
    time_stats = load_time_stats(args)

    # plot statistics
    plot_statistics(args, time_stats, fnc_stats)


if __name__ == "__main__":
    main()

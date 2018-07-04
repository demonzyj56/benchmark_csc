#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Image reconstruction using CSC."""
# pylint: disable=unused-import
import argparse
import datetime
import logging
import os
import sys
import pickle
import pprint
import subprocess
import pyfftw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import sporco.util as su
import sporco.metric as sm
from sporco.admm.cbpdn import ConvBPDN
from config import cfg, merge_cfg_from_file, merge_cfg_from_list
import image_dataset
from admm_slice import *


def setup_solver(solver_name, D, image):
    """Automatically import and setup solver with solver_name."""
    opt = globals()[solver_name].Options(dict(getattr(cfg, solver_name)))
    for k, v in cfg.cbpdn.items():
        opt[k] = v
    solver = globals()[solver_name](D, image, cfg.lmbda, opt)
    return solver


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


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description='Argument parser for image reconstruction using CSC'
    )
    parser.add_argument(
        '--print-cfg',
        dest='print_cfg',
        help='Print all config entries and exit',
        action='store_true'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training',
        default=None,
        type=str
    )
    parser.add_argument(
        'opts',
        help='Set arguments on command line',
        default=None,
        nargs=argparse.REMAINDER
    )

    return parser.parse_args()


def get_dict(gray):
    """Returns the pretrained dictionary."""
    # TODO(leoyolo): consider which dictionary to use
    if gray:
        D = su.convdicts()['G:8x8x64']
    else:
        D = su.convdicts()['RGB:8x8x3x64']
    return D


def train_model():
    """Train all the selected model."""
    logger = logging.getLogger(__name__)
    gray = 'gray' in cfg.dataset or cfg.rgb2gray
    logger.info('Using %s dataset', 'grayscale' if gray else 'color')
    D = get_dict(gray)
    img = image_dataset.create_image_blob(cfg.dataset, cfg.data_type,
                                          scaled=True, gray=gray)
    sl, sh = su.tikhonov_filter(img, 10., 16)
    solvers = [setup_solver(s, D.copy(), sh.copy()) for s in cfg.solvers]
    for idx, (solver, solver_name) in enumerate(zip(solvers, cfg.solvers)):
        logger.info(
            'Solving %d/%d CSC algorithm: %s', idx+1, len(solvers), solver_name
        )
        solver.solve()
        # reconstruction
        shr = solver.reconstruct().reshape(sh.shape)
        imgr = sl + shr
        logger.info(
            'Reconstruction PSNR for %s: %.2fdB',
            solver_name, sm.psnr(img, imgr)
        )
    for stat_name in cfg.statistics:
        filename = os.path.join(cfg.output_path, '{}.pdf'.format(stat_name))
        plot_statistics(solvers, stat_name, filename)

    return solvers


def plot_statistics(solvers, stat_name, filename):
    """Plot the iteration statistics stat_name and save to filename."""
    fig = plt.figure()
    for solver in solvers:
        plt.plot(
            solver.getitstat().Iter,
            getattr(solver.getitstat(), stat_name),
            label=type(solver).__name__
        )
    # set xtick to be integers
    # https://www.scivision.co/matplotlib-force-integer-labeling-of-axis/
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    fig.savefig(filename, bbox_inches='tight')


def main():
    """Setup everything for training."""
    args = parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    if args.print_cfg:
        pprint.pprint(cfg)
        return
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)
    log_name = os.path.join(
        cfg.output_path,
        '{:s}.{:%Y-%m-%d_%H-%M-%S}.log'.format(
            cfg.name,
            datetime.datetime.now(),
        )
    )
    # setup logger
    logger = setup_logging(__name__, log_name)
    logger.info('*' * 49)
    logger.info('* DATE: %s', str(datetime.datetime.now()))
    logger.info('*' * 49)
    logger.info('Called with args:')
    logger.info(args)
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    if cfg.rng_seed >= 0:
        np.random.seed(cfg.rng_seed)
        torch.manual_seed(cfg.rng_seed)
    if cfg.data_type == 'float32':
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)
    model = train_model()


if __name__ == "__main__":
    main()

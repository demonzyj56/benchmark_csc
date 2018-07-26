#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark code for dictlrn.

Dataset: fruit

Target impl: ConvBPDNSliceDictLearnFISTA

Reference impl: ConvBPDNDictLearn

"""
import argparse
import datetime
import logging
import os
import sys
import pyfftw  # pylint: disable=unused-import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sporco.util as su
from sporco.dictlrn.cbpdndl import ConvBPDNDictLearn
from sporco.dictlrn.prlcnscdl import ConvBPDNDictLearn_Consensus
from fista_slice import ConvBPDNSliceDictLearnFISTA
import image_dataset


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Options for benchmarking dictlrn algorithms CDL.')
    parser.add_argument('--name', default='benchmark_dictlrn', type=str, help='Name for experiment')
    parser.add_argument('--output_path', default='.default_dictlrn', type=str, help='Path for output')
    parser.add_argument('--lambda', '--lmbda', dest='lmbda', default=0.2, type=float, help='Lambda value for CDL')
    parser.add_argument('--xaxis', default='Iter', type=str, help='X-axis for statistics plot')
    parser.add_argument('--yaxis', default='ObjFun', type=str, help='Y-axis for statistics plot')
    return parser.parse_args()


def setup_ConvBPDNSliceDictLearnFISTA(D0, image, lmbda):
    """Setup target solver."""
    opt = ConvBPDNSliceDictLearnFISTA.Options({
        'Verbose': True,
        'MaxMainIter': 50,
        'CBPDN': {'MaxMainIter': 1, 'L': 100., 'BackTrack': {'Enabled': False}},
        'CCMOD': {'MaxMainIter': 1, 'L': 100., 'BackTrack': {'Enabled': False}},
        'AccurateDFid': True
    })
    return ConvBPDNSliceDictLearnFISTA(D0, image, lmbda, opt)


def setup_ConvBPDNDictLearn(D0, image, lmbda):
    """Setup reference solver."""
    opt = ConvBPDNDictLearn.Options({
        'Verbose': True,
        'MaxMainIter': 50,
        'CBPDN': {'rho': 50.*lmbda + 0.5},
        'CCMOD': {'rho': 10.0, 'ZeroMean': True},
        'AccurateDFid': True,
    }, dmethod='cns')
    return ConvBPDNDictLearn(D0, image, lmbda, opt)


def setup_ConvBPDNDictLearn_FISTA(D0, image, lmbda):
    opt = ConvBPDNDictLearn.Options({
        'Verbose': True,
        'MaxMainIter': 50,
        'CBPDN': {'BackTrack': {'Enabled': False}, 'L': 100.},
        'CCMOD': {'BackTrack': {'Enabled': False}, 'L': 100.},
        'AccurateDFid': True,
    }, xmethod='fista', dmethod='fista')
    return ConvBPDNDictLearn(D0, image, lmbda, opt)


def setup_ConvBPDNDictLearn_Consensus(D0, image, lmbda):
    """Setup parallel reference solver."""
    opt = ConvBPDNDictLearn_Consensus.Options({
        'Verbose': True,
        'MaxMainIter': 50,
        'CBPDN': {'rho': 50*lmbda + 0.5},
        'CCMOD': {'rho': 1.0, 'ZeroMean': True},
        'AccurateDFid': True
    })
    return ConvBPDNDictLearn_Consensus(D0, image, lmbda, opt)


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


def main():
    """Main entry."""
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    log_name = os.path.join(
        args.output_path,
        '{:s}.{:%Y-%m-%d_%H-%M-%S}.log'.format(
            args.name,
            datetime.datetime.now(),
        )
    )
    logger = setup_logging(__name__, log_name)
    train_data = image_dataset.create_image_blob('fruit', np.float32,
                                                 scaled=True, gray=False)
    sl, sh = su.tikhonov_filter(train_data, 5, 16)
    D0 = np.random.randn(8, 8, 3, 64)
    solvers = [globals()['setup_{}'.format(s)](D0, sh, args.lmbda) for s in
               ('ConvBPDNDictLearn_FISTA', 'ConvBPDNSliceDictLearnFISTA')]
    for s in solvers:
        logger.info('Solving CDL algorithm: %s', type(s).__name__)
        D1 = s.solve()
        logger.info('%s solve time: %.2fs', type(s).__name__,
                    s.timer.elapsed('solve'))
        np.save(os.path.join(args.output_path,
                             '{}_dict.npy'.format(type(s).__name__)),
                D1.squeeze())
    fig = plt.figure()
    for s in solvers:
        plt.plot(
            getattr(s.getitstat(), args.xaxis),
            getattr(s.getitstat(), args.yaxis),
            label=type(s).__name__
        )
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    fig.savefig(os.path.join(args.output_path,
                             '{}_{}.pdf'.format(args.xaxis, args.yaxis)),
                bbox_inches='tight')


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for online convolutional sparse coding."""
import argparse
import datetime
import glob
import logging
import pickle
import pprint
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
from sporco.admm import parcbpdn
from utils import setup_logging
from datasets import get_dataset, BlobLoader


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
    parser.add_argument('--use_gray', action='store_true', help='Use grayscale image instead of color image')
    return parser.parse_args()


def get_stats_from_dict(Dd, args, test_blob):
    """Test function for obtaining functional values and PSNR for each
    data point at test set. This only works for blob data.

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
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    logger = logging.getLogger(__name__)
    opt = yaml.load(open(args.cfg, 'r'))
    opt = cbpdn.ConvBPDN.Options(opt.get('ConvBPDN', None))
    #  opt = parcbpdn.ParConvBPDN.Options(opt.get('ConvBPDN', None))
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
            #  solver = parcbpdn.ParConvBPDN(D, sh, args.lmbda, opt=opt)
            solver.solve()
            fnc = solver.getitstat().ObjFun[-1]
            shr = solver.reconstruct().squeeze()
            imgr = sl + shr
            psnr = 0.
            for jdx in range(test_blob.shape[-1]):
                psnr += sm.psnr(test_blob[..., jdx], imgr[..., jdx], rng=1.)
            psnr /= test_blob.shape[-1]
            stats.append((fnc, psnr))
            del solver, imgr
            logger.info('%s %d/%d: ObjFun: %.2f, PSNR: %.2f',
                        name, idx+1, len(Ds), fnc, psnr)
        model_stats.update({name: stats})
    path = os.path.join(args.output_path, f'{dname}.final_stats.pkl')
    logger.info('Saving final statistics to %s', path)
    pickle.dump(model_stats, open(path, 'wb'))
    return model_stats


def load_dict(args):
    """Load dictionaries from output path."""
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    cfg = yaml.load(open(args.cfg, 'r'))
    del cfg['ConvBPDN']
    Dd = {}
    for k in cfg.keys():
        # load all dicts
        p = os.path.join(args.output_path, k)
        assert os.path.exists(p)
        npy_path = os.path.join(p, '{:s}.*.npy'.format(dname))
        npy_path = glob.glob(npy_path)
        assert all([os.path.exists(pp) for pp in npy_path])
        # sort paths according to index
        npy_path = sorted(npy_path, key=lambda t: int(t.split('.')[-2]))
        Ds = [np.load(pp) for pp in npy_path]
        Dd.update({k: Ds})
    return Dd


def load_time_stats(args):
    """Load time statistics from output path."""
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    cfg = yaml.load(open(args.cfg, 'r'))
    del cfg['ConvBPDN']
    time_stats = {}
    for k in cfg.keys():
        p = os.path.join(args.output_path, k, f'{dname}.time_stats.pkl')
        if os.path.exists(p):
            s = pickle.load(open(p, 'rb'))
            time_stats.update({k: s['Time']})
        else:
            p = os.path.join(args.output_path, k, f'{dname}.stats.npy')
            assert os.path.exists(p)
            s = np.load(p)
            time_stats.update({k: s[0][s[1].index('Time')]})
    return time_stats


def plot_statistics(args, time_stats=None, fnc_stats=None, class_legend=None):
    """Plot obtained statistics."""
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    if time_stats is None:
        time_stats = load_time_stats(args)
    if fnc_stats is None:
        p = os.path.join(args.output_path, f'{dname}.final_stats.npy')
        fnc_stats = np.load(p)

    fncs, psnrs = {}, {}
    for k, v in fnc_stats.items():
        fnc, psnr = list(zip(*v))
        fncs.update({k: fnc})
        psnrs.update({k: psnr})

    # plot objective value
    for k, v in time_stats.items():
        if class_legend is not None:
            label = class_legend.get(k, k)
        else:
            label = k
        plt.plot(v, fncs[k], label=label)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Test set objective')
    plt.savefig(os.path.join(args.output_path, f'{dname}.ObjFun.pdf'),
                bbox_inches='tight')
    plt.show()

    # plot psnr
    for k, v in time_stats.items():
        if class_legend is not None:
            label = class_legend.get(k, k)
        else:
            label = k
        plt.plot(v, psnrs[k], label=label)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('PSNR (dB)')
    plt.savefig(os.path.join(args.output_path, f'{dname}.PSNR.pdf'),
                bbox_inches='tight')
    plt.show()


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

    test_blob = get_dataset(args.dataset, train=False, gray=args.use_gray)

    # load all snapshots of dictionaries
    Dd = load_dict(args)

    # compute functional values and PSNRs from dictionaries
    fnc_stats = get_stats_from_dict(Dd, args, test_blob)

    # load time statistics
    time_stats = load_time_stats(args)

    # plot statistics
    plot_statistics(args, time_stats, fnc_stats)


if __name__ == "__main__":
    main()

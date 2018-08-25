#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for online convolutional sparse coding."""
import argparse
import datetime
import logging
import pickle
import pprint
import os
import re
import sys
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import sporco.util as su
import sporco.metric as sm
import sporco.linalg as spl
from sporco.admm import cbpdn, parcbpdn
import sporco_cuda.cbpdn as cucbpdn
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
    parser.add_argument('--use_gpu', action='store_true', help='Whether to use gpu impl of CBPDN to compute')
    parser.add_argument('--pad_boundary', action='store_true', help='Pad the boundary of test_blob with zeros')
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


def get_stats_from_dict_gpu(Dd, args, test_blob):
    """Test function for obtaining functional values and PSNR for each
    data point at test set. This only works for gray scale blob data, using
    gpu implementation.

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
    assert args.use_gray, 'Only grayscale image is supported'
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    logger = logging.getLogger(__name__)
    opt = yaml.load(open(args.cfg, 'r'))
    opt = cbpdn.ConvBPDN.Options(opt.get('ConvBPDN', None))
    if getattr(args, 'pad_boundary', False):
        dummy = list(Dd.values())[0][0]
        Hc, Wc = dummy.shape[:2]
        assert Hc % 2 == 1 and Wc % 2 == 1
        pad = [(Hc//2, Hc//2), (Wc//2, Wc//2)] + [(0, 0) for _ in range(test_blob.ndim-2)]
        test_blob = np.pad(test_blob, pad, 'constant')
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
            fnc = 0.
            psnr = 0.
            for jdx in range(test_blob.shape[-1]):
                X = cucbpdn.cbpdn(D, sh[..., jdx].squeeze(), args.lmbda, opt=opt)
                shr = np.sum(spl.fftconv(D, X), axis=2)
                dfd = linalg.norm(shr.ravel()-sh[..., jdx].ravel())**2 / 2.
                rl1 = linalg.norm(X.ravel(), 1)
                obj = dfd + args.lmbda * rl1
                fnc += obj
                imgr = sl + shr
                if getattr(args, 'pad_boundary', False):
                    imgr = imgr[Hc//2:-(Hc//2), Wc//2:-(Wc//2), ...]
                    img = test_blob[Hc//2:-(Hc//2), Wc//2:-(Wc//2), jdx]
                    psnr += sm.psnr(imgr, img, rng=1.)
                else:
                    psnr += sm.psnr(imgr, test_blob[..., jdx].squeeze(), rng=1.)
            psnr /= test_blob.shape[-1]
            stats.append((fnc, psnr))
            logger.info('%s %d/%d: ObjFun: %.2f, PSNR: %.2f',
                        name, idx+1, len(Ds), fnc, psnr)
        model_stats.update({name: stats})
    path = os.path.join(args.output_path, f'{dname}.final_stats.pkl')
    logger.info('Saving final statistics to %s', path)
    pickle.dump(model_stats, open(path, 'wb'))
    return model_stats


def load_dict(args):
    """Load dictionaries from output path."""
    #  dname = args.dataset if not args.use_gray else args.dataset+r'\.gray'
    # NOTE: hack, to support the case where train/test datasets are different.
    dname = getattr(args, 'dname', args.dataset)
    if args.use_gray:
        dname = dname + r'\.gray'
    cfg = yaml.load(open(args.cfg, 'r'))
    del cfg['ConvBPDN']
    Dd = {}
    for k in cfg.keys():
        # load all dicts
        try:
            base_dir = os.path.join(args.output_path, k)
            files = os.listdir(base_dir)
        except:
            logging.getLogger(__name__).info(
                'Folder %s does not exist, skipping...', base_dir
            )
            continue
        pattern = re.compile(r'{:s}\.[0-9]+\.npy'.format(dname))
        files = [f for f in files if pattern.match(f) is not None]
        npy_path = [os.path.join(base_dir, f) for f in files]
        assert all([os.path.exists(pp) for pp in npy_path])
        # sort paths according to index
        npy_path = sorted(npy_path, key=lambda t: int(t.split('.')[-2]))
        Ds = [np.load(pp) for pp in npy_path]
        Dd.update({k: Ds})
    return Dd


def load_time_stats(args):
    """Load time statistics from output path."""
    #  dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    # NOTE: hack, to support the case where train/test datasets are different.
    dname = getattr(args, 'dname', args.dataset)
    if args.use_gray:
        dname = dname + '.gray'
    cfg = yaml.load(open(args.cfg, 'r'))
    del cfg['ConvBPDN']
    time_stats = {}
    for k in cfg.keys():
        if not os.path.exists(os.path.join(args.output_path, k)):
            logging.getLogger(__name__).info(
                'Folder %s does not exist, skipping...',
                os.path.join(args.output_path, k)
            )
            continue
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
    #  dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    # NOTE: hack, to support the case where train/test datasets are different.
    dname = getattr(args, 'dname', args.dataset)
    if args.use_gray:
        dname = dname + '.gray'
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

    # plot objective value with respect to iterations
    for k, v, in fncs.items():
        if class_legend is not None:
            label = class_legend.get(k, k)
        else:
            label = k
        plt.plot(v, label=label)
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Test set objective')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(os.path.join(args.output_path, f'{dname}.ObjFun2.pdf'),
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
    if not args.use_gpu:
        fnc_stats = get_stats_from_dict(Dd, args, test_blob)
    else:
        fnc_stats = get_stats_from_dict_gpu(Dd, args, test_blob)

    # load time statistics
    time_stats = load_time_stats(args)

    # plot statistics
    plot_statistics(args, time_stats, fnc_stats)


if __name__ == "__main__":
    main()

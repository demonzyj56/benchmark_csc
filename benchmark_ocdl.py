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
from onlinecdl_sgd import OnlineDictLearnSGD
from fista_slice_online import OnlineDictLearnSliceSurrogate
from onlinecdl_surrogate import OnlineDictLearnDenseSurrogate
from sporco.dictlrn import prlcnscdl, cbpdndl
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
    parser.add_argument('--pad_boundary', action='store_true', help='Pad the input blob boundary with zeros')
    parser.add_argument('--visdom', default=None, help='Whether should setup a visdom server')
    parser.add_argument('--batch_model', action='store_true', help='Optionally train the batch dictlrn model')
    parser.add_argument('--no_legacy', action='store_true', help='Remove personal legacy code for research use')
    return parser.parse_args()


def train_models(D0, solvers, train_blob, args):
    """Function for training every solvers."""
    loader = BlobLoader(train_blob, args.epochs, args.batch_size)
    sample = loader.random_sample()
    solvers = {k: sol_name(D0, sample, args.lmbda, opt=opt) for
               k, (sol_name, opt) in solvers.items()}
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    for e, blob in enumerate(loader):
        if args.pad_boundary:
            assert args.patch_size % 2 == 1, 'Patch size should be odd'
            radius = args.patch_size // 2
            pad = [(radius, radius), (radius, radius)] + \
                [(0, 0) for _ in range(blob.ndim-2)]
            #  pad = [(0, args.patch_size-1), (0, args.patch_size-1)] + \
            #      [(0, 0) for _ in range(blob.ndim-2)]
            blob = np.pad(blob, pad, mode='constant')
        if not args.no_tikhonov_filter:
            # fix lambda to be 5
            _, blob = su.tikhonov_filter(blob, 5.)
        for k, s in solvers.items():
            s.solve(blob.copy())
            path = os.path.join(args.output_path, k)
            np.save(os.path.join(path, '{}.{}.npy'.format(dname, e)),
                    s.getdict().squeeze())
            if args.visdom is not None:
                tiled_dict = su.tiledict(s.getdict().squeeze())
                if not args.use_gray:
                    tiled_dict = tiled_dict.transpose(2, 0, 1)
                args.visdom.image(tiled_dict, opts=dict(caption=f'{k}.{e}'))
    # snapshot iteration record
    sio.savemat(os.path.join(args.output_path, 'iter_record.mat'),
                {'iter_record': loader.record})
    return solvers


def train_batch_model(D0, train_blob, opt, args):
    """Train for batch dictlrn model."""
    logger = logging.getLogger(__name__)
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    if os.path.exists(os.path.join(args.output_path, 'iter_record.mat')):
        iter_record = sio.loadmat(os.path.join(args.output_path, 'iter_record.mat'))['iter_record']
        selected = list(set(iter_record.ravel().tolist()))
        if len(selected) < train_blob.shape[-1]:
            train_blob = train_blob[..., selected]
            logger.info('Selected %d -> %d train samples for training batch model')
    if not args.no_tikhonov_filter:
        # fix lambda to be 5
        _, train_blob = su.tikhonov_filter(train_blob, 5.)
    path = os.path.join(args.output_path, 'ConvBPDNDictLearn')
    if not os.path.exists(path):
        os.makedirs(path)

    def _callback(d):
        """Snapshot dictionaries for every iteration."""
        _D = d.getdict().squeeze()
        np.save(os.path.join(path, '{}.{}.npy'.format(dname, d.j)), _D)
        return 0

    opt['Callback'] = _callback
    solver = cbpdndl.ConvBPDNDictLearn(D0, train_blob, args.lmbda, opt=opt,
                                       xmethod='admm', dmethod='cns')
    solver.solve()
    return solver


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
            args.dataset if not args.use_gray else args.dataset+'.gray',
            datetime.datetime.now(),
        )
    )
    logger = setup_logging(__name__, log_name)
    logger.info('args:')
    logger.info(pprint.pformat(args))

    # setup visdom server
    if args.visdom is not None:
        import visdom
        args.visdom = visdom.Visdom()

    # set random seed
    if args.rng_seed >= 0:
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)

    # load configs
    cfg = yaml.load(open(args.cfg, 'r'))
    if 'ConvBPDN' in cfg.keys():
        del cfg['ConvBPDN']
    logger.info('cfg:')
    logger.info(pprint.pformat(cfg))

    # train batch model optionally
    if args.batch_model:
        if 'ConvBPDNDictLearn' in cfg.keys():
            bopt = cbpdndl.ConvBPDNDictLearn.Options(cfg['ConvBPDNDictLearn'],
                                                     xmethod='admm',
                                                     dmethod='cns')
        else:
            bopt = cbpdndl.ConvBPDNDictLearn.Options(xmethod='admm',
                                                     dmethod='cns')
    if 'ConvBPDNDictLearn' in cfg.keys():
        del cfg['ConvBPDNDictLearn']

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

    if args.use_gray:
        D0 = np.random.randn(args.patch_size, args.patch_size, args.num_atoms).astype(np.float32)
        D0[..., 0] = np.ones((args.patch_size, args.patch_size), dtype=np.float32)/(args.patch_size**2)
    else:
        D0 = np.random.randn(args.patch_size, args.patch_size, 3, args.num_atoms).astype(np.float32)
        D0[..., 0] = np.ones((args.patch_size, args.patch_size, 3), dtype=np.float32)/(3*args.patch_size**2)

    # train solvers
    solvers = train_models(D0, solvers, train_blob, args)

    # train batch model optionally
    if args.batch_model:
        solver = train_batch_model(D0, train_blob, bopt, args)
        solvers.update(dict(ConvBPDNDictLearn=solver))

    # plot and save everything
    plot_and_save_statistics(solvers, args)

    # legacy code for research
    if not args.no_legacy:
        if args.dataset == 'city' or args.dataset == 'fruit':
            os.system('cp {:s}/iter_record.mat /backup/OCSC/datasets/{:s}_10/'.format(args.output_path, args.dataset))  # XXX


if __name__ == "__main__":
    main()

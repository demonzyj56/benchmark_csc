#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script for VOC07, tested with filters learned from noisy data."""
import argparse
import copy
import datetime
import pickle
import pprint
import os
import sys
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
import torch
import torch.utils.data as tdata
from sporco.admm import cbpdn
import sporco.util as su
import sporco.metric as sm
from utils import logging, setup_logging
from datasets import get_dataset, BlobLoader
from test_ocdl import load_dict, load_time_stats, plot_statistics


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Options for benchmarking dictlrn algorithms CDL.')
    parser.add_argument('--name', default='ocdl', type=str, help='Name for experiment')
    parser.add_argument('--train_dataset', dest='dname', default='voc07', type=str, help='Hack to use different train/test set')
    parser.add_argument('--output_path', default='.voc07.mask', type=str, help='Path for output')
    parser.add_argument('--dataset', default='voc07', type=str, help='Dataset to use')
    parser.add_argument('--lambda', '--lmbda', dest='lmbda', default=0.2, type=float, help='Lambda value for CDL')
    parser.add_argument('--cfg', default='experiments/voc07.mask.yml', type=str, help='Config yaml file for solvers')
    parser.add_argument('--rng_seed', default=-1, type=int, help='Random seed to use; negative values mean dont set')
    parser.add_argument('--no_tikhonov_filter', default=False, type=bool, help='No tikhonov low pass filtering is applied')
    parser.add_argument('--use_gray', action='store_true', help='Use grayscale image instead of color image')
    parser.add_argument('--last_only', action='store_true', help='Only compute the statistics from last epoch')
    parser.add_argument('--pad_boundary', default=False, type=bool, help='Pad the boundary of test_blob with zeros')
    parser.add_argument('--num_test', default=20, type=int, help='Number of test samples; negative values mean use all')
    return parser.parse_args()


class SubSet(tdata.Dataset):
    """Ported from latest pytorch code."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def get_stats_from_dict(Dd, args, test_blob):
    """Loop over given images one by one."""
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    logger = logging.getLogger(__name__)
    opt = yaml.load(open(args.cfg, 'r'))
    opt = cbpdn.ConvBPDN.Options(opt.get('ConvBPDN', None))
    model_stats = {}
    for name, Ds in Dd.items():
        logger.info('Testing solver %s', name)
        stats = []
        for idx, D in enumerate(Ds):
            loader = BlobLoader(test_blob)
            fnc, psnr = 0., 0.
            for jdx, blob in enumerate(loader):
                if not args.no_tikhonov_filter:
                    sl, sh = su.tikhonov_filter(blob, 5)
                else:
                    sl, sh = 0., blob
                solver = cbpdn.ConvBPDN(D, sh, args.lmbda, opt=opt)
                solver.solve()
                ff = solver.getitstat().ObjFun[-1]
                fnc += ff
                shr = solver.reconstruct()
                imgr = sl + shr.reshape(sl.shape)
                pp = sm.psnr(blob, imgr, rng=1.)
                psnr += pp
                logger.info('Image %d/%d, ObjFun: %.2f, PSNR: %.2f',
                            jdx+1, len(loader), ff, pp)
            psnr /= len(loader)
            stats.append((fnc, psnr))
            logger.info('%s %d/%d: ObjFun: %.2f, PSNR: %.2f',
                        name, idx+1, len(Ds), fnc, psnr)
        model_stats.update({name: stats})
    path = os.path.join(args.output_path, '{}.final_stats.pkl'.format(dname))
    logger.info('Saving final statistics to %s', path)
    pickle.dump(model_stats, open(path, 'wb'))
    return model_stats


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
    else:
        np.random.seed(None)

    test_blob = get_dataset(args.dataset, train=False, gray=args.use_gray)
    # sample test set
    if args.num_test > 0 and args.num_test < len(test_blob):
        index_name = os.path.join(args.output_path,
                                  f'{args.dataset}.test_idx.{args.num_test}.npy')
        if os.path.exists(index_name):
            selected = np.load(index_name)
        else:
            selected = np.random.choice(len(test_blob), args.num_test,
                                        replace=False)
            np.save(index_name, selected)
        logger.info('Selecting %d -> %d test samples from %s.',
                    len(test_blob), args.num_test, args.dataset)
        test_blob = SubSet(test_blob, selected)

    Dd = load_dict(args)

    fnc_stats = get_stats_from_dict(Dd, args, test_blob)

    # load time statistics
    time_stats = load_time_stats(args)

    # plot statistics
    plot_statistics(args, time_stats, fnc_stats)


if __name__ == "__main__":
    main()

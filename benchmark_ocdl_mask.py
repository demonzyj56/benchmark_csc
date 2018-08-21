#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark code for online convolutional sparse coding with masking."""
import argparse
import datetime
import os
import pprint
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
import scipy.io as sio
import torch
from sporco.admm import tvl2
from sporco.dictlrn import cbpdndlmd
import sporco.util as su
from onlinecdl_sgd import OnlineDictLearnSGDMask
from fista_slice_online import OnlineDictLearnSliceSurrogate
from datasets import get_dataset, BlobLoader
from utils import setup_logging
from benchmark_ocdl import plot_and_save_statistics


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Options for benchmarking online CDL with mask.')
    parser.add_argument('--name', default='ocdl_mask', type=str, help='Name for experiment')
    parser.add_argument('--output_path', default='.default_ocdl_mask', type=str, help='Path for output')
    parser.add_argument('--dataset', default='voc07', type=str, help='Dataset to use')
    parser.add_argument('--lambda', '--lmbda', dest='lmbda', default=0.1, type=float, help='Lambda value for CDL')
    parser.add_argument('--l2_lambda', default=0.05, type=float, help='Lambda for L2-denoising')
    parser.add_argument('--noise_fraction', default=0.5, type=float, help='The fraction of pixels that are corrupted (for each channel)')
    parser.add_argument('--epochs', default=40, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of samples for each epoch')
    parser.add_argument('--cfg', default='experiments/voc07.mask.yml', type=str, help='Config yaml file for solvers')
    parser.add_argument('--patch_size', default=8, type=int, help='Height and width for dictionary')
    parser.add_argument('--num_atoms', default=25, type=int, help='Size of dictionary')
    parser.add_argument('--rng_seed', default=-1, type=int, help='Random seed to use; negative values mean dont set')
    parser.add_argument('--use_gray', action='store_true', help='Use grayscale image instead of color image')
    parser.add_argument('--dont_pad_boundary', action='store_true', help='Do not pad the input blob boundary with zeros')
    parser.add_argument('--visdom', default=None, help='Whether should setup a visdom server')
    parser.add_argument('--batch_model', action='store_true', help='Optionally train the batch dictlrn model')
    return parser.parse_args()


def train_models(solvers, train_loader, args):
    """Train for all solvers."""
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    masks = []
    shs = []
    for e, blob in enumerate(train_loader):
        mask = su.rndmask(blob.shape, args.noise_fraction, dtype=blob.dtype)
        blobw = blob * mask
        if not args.dont_pad_boundary:
            pad = [(0, args.patch_size-1), (0, args.patch_size-1)] + \
                [(0, 0) for _ in range(blob.ndim-2)]
            blobw = np.pad(blobw, pad, 'constant')
            mask = np.pad(mask, pad, 'constant')
        # l2-TV denoising
        tvl2opt = tvl2.TVL2Denoise.Options({
            'Verbose': False, 'MaxMainIter': 200, 'gEvalY': False,
            'AutoRho': {'Enabled': True}, 'DFidWeight': mask
        })
        denoiser = tvl2.TVL2Denoise(blobw, args.l2_lambda, tvl2opt,
                                    caxis=None if args.use_gray else 2)
        sl = denoiser.solve()
        sh = mask * (blobw - sl)
        # save masks and sh
        masks.append(mask)
        shs.append(sh)
        # Update solvers
        for k, solver in solvers.items():
            solver.solve(sh, W=mask)
            np.save(os.path.join(args.output_path, k, '{}.{}.npy'.format(dname, e)),
                    solver.getdict().squeeze())
            if args.visdom is not None:
                tiled_dict = su.tiledict(solver.getdict().squeeze())
                if not args.use_gray:
                    tiled_dict = tiled_dict.transpose(2, 0, 1)
                args.visdom.image(tiled_dict, opts=dict(caption=f'{k}.{e}'))
    # snapshot blobs and masks
    masks = np.concatenate(masks, axis=-1)
    shs = np.concatenate(shs, axis=-1)
    np.save(os.path.join(args.output_path, 'train_masks.npy'), masks)
    np.save(os.path.join(args.output_path, 'train_blobs.npy'), shs)

    return solvers, shs, masks


def train_batch_model(args, D0, train_blobs=None, masks=None):
    """Train a batch CDL with spatial masking."""
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    path = os.path.join(args.output_path, 'ConvBPDNMaskDictLearn')
    if not os.path.exists(path):
        os.makedirs(path)
    if train_blobs is None:
        train_blobs = np.load(os.path.join(args.output_path, 'train_blobs.npy'))
    if masks is None:
        masks = np.load(os.path.join(args.output_path, 'train_masks.npy'))
    cfg = yaml.load(open(args.cfg, 'r'))
    opts = cfg.get('ConvBPDNMaskDictLearn', {})

    def _callback(d):
        """Snapshot dictionaries for every iteration."""
        _D = d.getdict().squeeze()
        np.save(os.path.join(path, '{}.{}.npy'.format(dname, d.j)), _D)
        if args.visdom is not None:
            tiled_dict = su.tiledict(_D)
            if not args.use_gray:
                tiled_dict = tiled_dict.transpose(2, 0, 1)
            args.visdom.image(tiled_dict, opts=dict(caption='ConvBPDNMaskDictLearn.{}'.format(d.j)))
        return 0

    opts.update({'Callback': _callback})
    mdopt = cbpdndlmd.ConvBPDNMaskDictLearn.Options(opts, dmethod='cns')
    solver = cbpdndlmd.ConvBPDNMaskDictLearn(D0, train_blobs, args.lmbda, masks,
                                             opt=mdopt, dmethod='cns')
    solver.solve()
    return solver


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
        args.visdom = visdom.Visdom(env='benchmark_ocdl_mask')

    # set random seed
    if args.rng_seed >= 0:
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)
    else:
        np.random.seed(None)

    # load configs
    cfg = yaml.load(open(args.cfg, 'r'))
    if 'ConvBPDN' in cfg.keys():
        del cfg['ConvBPDN']
    if 'ConvBPDNMaskDictLearn' in cfg.keys():
        del cfg['ConvBPDNMaskDictLearn']
    logger.info('cfg:')
    logger.info(pprint.pformat(cfg))

    train_blob = get_dataset(args.dataset, gray=args.use_gray, train=True)
    loader = BlobLoader(train_blob, args.epochs, args.batch_size)
    rand_sample = loader.random_sample()
    # optionally pad boundary for rand_sample
    if not args.dont_pad_boundary:
        pad = [(0, args.patch_size-1), (0, args.patch_size-1)] + \
            [(0, 0) for _ in range(rand_sample.ndim-2)]
        rand_sample = np.pad(rand_sample, pad, 'constant')

    if args.use_gray:
        D0 = np.random.randn(args.patch_size, args.patch_size, args.num_atoms).astype(np.float32)
    else:
        D0 = np.random.randn(args.patch_size, args.patch_size, 3, args.num_atoms).astype(np.float32)

    # load solvers
    solvers = {}
    for k, v in cfg.items():
        solver_class = globals()[k.split('-')[0]]
        path = os.path.join(args.output_path, k)
        if not os.path.exists(path):
            os.makedirs(path)
        opt = solver_class.Options(v)
        solvers.update({
            k: solver_class(D0, rand_sample, args.lmbda, opt=opt)
        })

    # train solvers
    solvers, shs, masks = train_models(solvers, loader, args)

    # train batch model optionally
    if args.batch_model:
        batch_solver = train_batch_model(args, D0, shs, masks)
        solvers.update({'ConvBPDNMaskDictLearn': batch_solver})

    plot_and_save_statistics(solvers, args)
    return solvers


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test OCDL with masking."""
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
import torch.utils.data as tdata
import sporco.util as su
import sporco.metric as sm
import sporco.linalg as spl
from sporco.admm import tvl2, cbpdn
from utils import setup_logging
from datasets import get_dataset, BlobLoader
from test_ocdl import load_dict, load_time_stats, plot_statistics


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Options for benchmarking dictlrn algorithms CDL.')
    parser.add_argument('--name', default='ocdl', type=str, help='Name for experiment')
    parser.add_argument('--train_dataset', dest='dname', default='voc07', type=str, help='Hack, indicate the train dataset')
    parser.add_argument('--output_path', default='.17flowers.mask', type=str, help='Path for output')
    parser.add_argument('--dataset', default='17flowers', type=str, help='Dataset to use')
    parser.add_argument('--lambda', '--lmbda', dest='lmbda', default=0.1, type=float, help='Lambda value for CDL')
    parser.add_argument('--l2_lambda', default=0.05, type=float, help='Lambda for L2-denoising')
    parser.add_argument('--cfg', default='experiments/17flowers.mask.yml', type=str, help='Config yaml file for solvers')
    parser.add_argument('--rng_seed', default=-1, type=int, help='Random seed to use; negative values mean dont set')
    parser.add_argument('--use_gray', action='store_true', help='Use grayscale image instead of color image')
    parser.add_argument('--use_gpu', action='store_true', help='Whether to use gpu impl of CBPDN to compute')
    parser.add_argument('--num_test', default=-1, type=int, help='Number of test samples; negative values mean use all')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch size of the dictionary')
    parser.add_argument('--noise_fraction', default=0.3, type=float, help='Percentage of noise to add on each image')
    parser.add_argument('--dont_pad_boundary', action='store_true', help='Do not pad the input blob boundary with zeros')
    parser.add_argument('--last_only', action='store_true', help='Only compute statistics of the last epoch')
    parser.add_argument('--visdom', default=None, help='Whether should setup a visdom server')
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


def create_dataset(name, gray, num_test):
    """Create the desire dataset, returns an iterator.
    Only valid for batches of color images."""
    test_blob = get_dataset(name, train=False, gray=gray)
    if isinstance(test_blob, np.ndarray):
        assert test_blob.ndim == 4, 'Only batches of color images are supported'
        test_blob = np.split(test_blob, test_blob.shape[-1], axis=-1)
        test_blob = [t.squeeze() for t in test_blob]
    if num_test > 0 and num_test < len(test_blob):
        logging.getLogger(__name__).info('Selecting %d -> %d test samples from %s.',
                                         len(test_blob), num_test, name)
        selected = np.random.choice(len(test_blob), num_test, replace=False)
        test_blob = SubSet(test_blob, selected)
    return iter(test_blob)


def create_batch(args, dataset):
    masks = []
    blobs = []
    sls = []
    shs = []
    for blob in dataset:
        blob = blob.squeeze()
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
        sh = blobw - sl
        # save masks and sh
        masks.append(mask)
        blobs.append(blob)
        sls.append(sl)
        shs.append(sh)
    # snapshot blobs and masks
    masks = np.stack(masks, axis=-1)
    blobs = np.stack(blobs, axis=-1)
    sls = np.stack(sls, axis=-1)
    shs = np.stack(shs, axis=-1)
    np.save(os.path.join(args.output_path, 'test_masks.npy'), masks)
    np.save(os.path.join(args.output_path, 'test_blobs.npy'), blobs)
    np.save(os.path.join(args.output_path, 'test_blobs_sl.npy'), sls)
    np.save(os.path.join(args.output_path, 'test_blobs_sh.npy'), shs)
    logging.getLogger(__name__).info('Saved test data')
    return masks, blobs, sls, shs


def get_stats_from_dict(Dd, args, masks, blobs, sls, shs):
    dname = args.dataset if not args.use_gray else args.dataset+'.gray'
    logger = logging.getLogger(__name__)
    opt = yaml.load(open(args.cfg, 'r'))
    #  opt = cbpdn.ConvBPDNMaskDcpl.Options(opt.get('ConvBPDN', None))
    opt = cbpdn.ConvBPDN.Options(opt.get('ConvBPDN', None))
    model_stats = {}
    imgw = sls + shs
    imgw = imgw[:blobs.shape[0], :blobs.shape[1], ...]
    for name, Ds in Dd.items():
        logger.info('Testing solver %s', name)
        stats = []
        for idx, D in enumerate(Ds):
            #  solver = cbpdn.ConvBPDNMaskDcpl(D, shs, args.lmbda, W=masks, opt=opt)
            solver = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, shs, masks, args.lmbda, opt=opt)
            solver.solve()
            shr = solver.reconstruct().squeeze()
            imgr = sls + shr.reshape(sls.shape)
            imgr = imgr[:blobs.shape[0], :blobs.shape[1], ...]
            psnr = 0.
            for jdx in range(blobs.shape[-1]):
                pp = sm.psnr(blobs[..., jdx], imgr[..., jdx], rng=1.)
                psnr += pp
                logger.info('PSNR for %d-th image: %.3f', idx+1, pp)
                if args.visdom is not None:
                    images = [blobs[..., jdx].squeeze(), imgw[..., jdx].squeeze(),
                              imgr[..., jdx].squeeze()]
                    if not args.use_gray:
                        images = [b.transpose(2, 0, 1) for b in images]
                    args.visdom.images(images, opts=dict(caption='{}.{}.{}'.format(idx, jdx, name), nrow=3))
            psnr /= blobs.shape[-1]
            fnc = solver.getitstat().ObjFun[-1]
            stats.append((fnc, psnr))
            logger.info('%s %d/%d: ObjFun: %.2f, PSNR: %.2f',
                        name, idx+1, len(Ds), fnc, psnr)
            if idx == len(Ds) - 1:  # last iteration
                imgr_path = os.path.join(args.output_path, name, '{}.reconstruction.npy'.format(dname))
                logger.info('Saving final reconstruction to %s', imgr_path)
                np.save(imgr_path, imgr)
            del solver, imgr
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
        '{:s}.{:s}.{:%Y-%m-%d_%H-%M-%S}.mask.test.log'.format(
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

    # setup visdom server
    if args.visdom is not None:
        import visdom
        args.visdom = visdom.Visdom(env='test_ocdl_mask')

    dataset = create_dataset(args.dataset, args.use_gray, args.num_test)

    masks, blobs, sls, shs = create_batch(args, dataset)
    Dd = load_dict(args)
    if args.last_only:
        Dd = {k: [v[-1]] for k, v in Dd.items()}

    fnc_stats = get_stats_from_dict(Dd, args, masks, blobs, sls, shs)

    time_stats = load_time_stats(args)

    # plot statistics
    if not args.last_only:
        plot_statistics(args, time_stats, fnc_stats)


if __name__ == "__main__":
    main()

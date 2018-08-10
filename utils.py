"""Utility functions."""
import logging
import sys
import numpy as np


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


def Pcn(D, zm=True):
    """Constraint set projection function that normalizes each dictionary
    to unary norm.

    Parameters
    ----------
    D: array
        Input dictionary to normalize.
    zm: bool
        If true, then the columns are mean subtracted before normalization.
    """
    # sum or average over all but last axis
    axis = tuple(range(D.ndim-1))
    if zm:
        D -= np.mean(D, axis=axis, keepdims=True)
    norm = np.sqrt(np.sum(D**2, axis=axis, keepdims=True))
    norm[norm == 0] = 1.
    return np.asarray(D / norm, dtype=D.dtype)


def Pcn2(D, zm=True):
    """Projection function to normalize each dictionary to have norm less
    or equal than one.
    """
    axis = tuple(range(D.ndim-1))
    if zm:
        D -= np.mean(D, axis=axis, keepdims=True)
    norm = np.sqrt(np.sum(D**2, axis=axis, keepdims=True))
    norm[norm == 0.] = 1.
    norm[norm < 1.] = 1.
    return np.asarray(D / norm, dtype=D.dtype)

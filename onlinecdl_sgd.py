# -*- coding: utf-8 -*-
# Copyright (C) 2018 by Cristina Garcia-Cardona <cgarciac@lanl.gov>
#                       Brendt Wohlberg <brendt@ieee.org>
# All rights reserved. BSD 3-clause License.
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.
#
# Code obtained from original SPORCO library (commit id: 44852c1).
# Class name change: OnlineConvBPDNDictLearn -> OnlineDictLearnSGD
# Modified by leoyolo 15/08/2018.
#
# Class name change: OnlineConvBPDNMaskDictLearn -> OnlineDictLearnSGDMask
# Inherit now from OnlineDictLearnSGD.
# Modified by leoyolo 18/08/2018.

"""Online dictionary learning based on CBPDN sparse coding"""

from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from builtins import object

import pyfftw
import copy
import numpy as np
from scipy import linalg

from sporco import util
from sporco import common
from sporco.util import u
import sporco.linalg as sl
import sporco.cnvrep as cr
from sporco.admm import cbpdn, parcbpdn
from sporco.dictlrn import dictlrn
from sporco_cuda import cbpdn as cucbpdn


__author__ = """\n""".join(['Cristina Garcia-Cardona <cgarciac@lanl.gov>',
                            'Brendt Wohlberg <brendt@ieee.org>'])



class OnlineDictLearnSGD(common.IterativeSolver):
    r"""

    Stochastic gradient descent (SGD) based online convolutional
    dictionary learning, as proposed in :cite:`liu-2018-first`.
    """


    class Options(dictlrn.DictLearn.Options):
        r"""Online CBPDN dictionary learning algorithm options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is
          displayed.

          ``StatusHeader`` : Flag determining whether status header and
          separator are displayed.

          ``IterTimer`` : Label of the timer to use for iteration times.

          ``DictSize`` : Dictionary size vector.

          ``DataType`` : Specify data type for solution variables,
          e.g. ``np.float32``.

          ``ZeroMean`` : Flag indicating whether the solution
          dictionary :math:`\{\mathbf{d}_m\}` should have zero-mean
          components.

          ``eta_a``, ``eta_b`` : Constants :math:`a` and :math:`b` used
          in setting the SGD step size, :math:`\eta`, which is set to
          :math:`a / (b + i)` where :math:`i` is the iteration index.
          See Sec. 3 (pg. 9) of :cite:`liu-2018-first`.

          ``CUDA_CBPDN`` : Flag indicating whether to use CUDA solver
          for CBPDN problem (see :ref:`cuda_package`)

          ``CBPDN`` : Options :class:`.admm.cbpdn.ConvBPDN.Options`.
        """

        defaults = {'Verbose': False, 'StatusHeader': True,
                    'IterTimer': 'solve', 'DictSize': None,
                    'DataType': None, 'ZeroMean': False, 'eta_a': 10.0,
                    'eta_b': 5.0, 'CUDA_CBPDN': False,
                    'PAR_CBPDN': False,
                    'CBPDN': copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)}


        def __init__(self, opt=None):
            """Initialise online CBPDN dictionary learning algorithm
            options.
            """

            dictlrn.DictLearn.Options.__init__(self, {
                'CBPDN': cbpdn.ConvBPDN.Options({
                    'AutoRho': {'Period': 10, 'AutoScaling': False,
                    'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}})
                })

            if opt is None:
                opt = {}
            self.update(opt)


    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    """Fields in IterationStats associated with the objective function"""
    itstat_fields_alg = ('PrimalRsdl', 'DualRsdl', 'Rho', 'Cnstr',
                         'DeltaD', 'Eta')
    """Fields in IterationStats associated with the specific solver
    algorithm"""
    itstat_fields_extra = ()
    """Non-standard fields in IterationStats; see :meth:`itstat_extra`"""



    def __new__(cls, *args, **kwargs):
        """Create an OnlineDictLearnSGD object and start its
        initialisation timer."""

        instance = super(OnlineDictLearnSGD, cls).__new__(cls)
        instance.timer = util.Timer(['init', 'solve', 'solve_wo_eval'])
        instance.timer.start('init')
        return instance



    def __init__(self, D0, S0=None, lmbda=None, opt=None, dimK=None, dimN=2):
        """Initialise an OnlineDictLearnSGD object with problem
        size and options.

        Parameters
        ----------
        D0 : array_like
          Initial dictionary array
        S0:
          Dummy, keep to compatible with older code
        lmbda : float
          Regularisation parameter
        opt : :class:`OnlineDictLearnSGD.Options` object
          Algorithm options
        dimK : 0, 1, or None, optional (default None)
          Number of signal dimensions in signal array passed to
          :meth:`solve`. If there will only be a single input signal
          (e.g. if `S` is a 2D array representing a single image)
          `dimK` must be set to 0.
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        if opt is None:
            opt = OnlineDictLearnSGD.Options()
        if not isinstance(opt, OnlineDictLearnSGD.Options):
            raise TypeError('Parameter opt must be an instance of '
                            'OnlineDictLearnSGD.Options')
        self.opt = opt

        if dimN != 2 and opt['CUDA_CBPDN']:
            raise ValueError('CUDA CBPDN solver can only be used when dimN=2')

        self.dimK = dimK
        self.dimN = dimN

        # DataType option overrides data type inferred from __init__
        # parameters of derived class
        self.set_dtype(opt, D0.dtype)

        # Initialise attributes representing algorithm parameter
        self.lmbda = lmbda
        self.eta_a = opt['eta_a']
        self.eta_b = opt['eta_b']
        self.set_attr('eta', opt['eta_a'] / opt['eta_b'],
                      dval=2.0, dtype=self.dtype)

        # Get dictionary size
        if self.opt['DictSize'] is None:
            self.dsz = D0.shape
        else:
            self.dsz = self.opt['DictSize']

        # Construct object representing problem dimensions
        self.cri = None

        # Normalise dictionary
        ds = cr.DictionarySize(self.dsz, dimN)
        dimCd = ds.ndim - dimN - 1
        D0 = cr.stdformD(D0, ds.nchn, ds.nflt, dimN).astype(self.dtype)
        self.D = cr.Pcn(D0, self.dsz, (), dimN, dimCd, crp=True,
                        zm=opt['ZeroMean'])
        self.Dprv = self.D.copy()

        # Create constraint set projection function
        self.Pcn = cr.getPcn(self.dsz, (), dimN, dimCd, crp=True,
                             zm=opt['ZeroMean'])

        # Initalise iterations stats list and iteration index
        self.itstat = []
        self.j = 0

        # Configure status display
        self.display_config()
        self.display_start()



    def solve(self, S, dimK=None):
        """Compute sparse coding and dictionary update for training
        data `S`."""

        # Use dimK specified in __init__ as default
        if dimK is None and self.dimK is not None:
            dimK = self.dimK

        # Start solve timer
        self.timer.start(['solve', 'solve_wo_eval'])

        # Solve CSC problem on S and do dictionary step
        self.init_vars(S, dimK)
        self.xstep(S, self.lmbda, dimK)
        self.dstep()

        # Stop solve timer
        self.timer.stop('solve_wo_eval')

        # Extract and record iteration stats
        self.manage_itstat()

        # Increment iteration count
        self.j += 1

        # Stop solve timer
        self.timer.stop('solve')

        # Return current dictionary
        return self.getdict()



    def init_vars(self, S, dimK):
        """Initalise variables required for sparse coding and dictionary
        update for training data `S`."""

        Nv = S.shape[0:self.dimN]
        if self.cri is None or Nv != self.cri.Nv:
            self.cri = cr.CDU_ConvRepIndexing(self.dsz, S, dimK, self.dimN)
            if self.opt['CUDA_CBPDN']:
                if self.cri.Cd > 1 or self.cri.Cx > 1:
                    raise ValueError('CUDA CBPDN solver can only be used for '
                                     'single channel problems')
                if self.cri.K > 1:
                    raise ValueError('CUDA CBPDN solver can not be used with '
                                     'mini-batches')
            #  self.Df = sl.pyfftw_byte_aligned(sl.rfftn(self.D, self.cri.Nv,
            #                                            self.cri.axisN))
            self.Df = pyfftw.byte_align(sl.rfftn(self.D, self.cri.Nv,
                                                 self.cri.axisN))
            self.Gf = sl.pyfftw_empty_aligned(self.Df.shape, self.Df.dtype)
            self.Z = sl.pyfftw_empty_aligned(self.cri.shpX, self.dtype)
        else:
            self.Df[:] = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)



    def xstep(self, S, lmbda, dimK):
        """Solve CSC problem for training data `S`."""

        if self.opt['CUDA_CBPDN']:
            Z = cucbpdn.cbpdn(self.D.squeeze(), S[..., 0], lmbda,
                              self.opt['CBPDN'])
            Z = Z.reshape(self.cri.Nv + (1, 1, self.cri.M,))
            self.Z[:] = np.asarray(Z, dtype=self.dtype)
            self.Zf = sl.rfftn(self.Z, self.cri.Nv, self.cri.axisN)
            self.Sf = sl.rfftn(S.reshape(self.cri.shpS), self.cri.Nv,
                               self.cri.axisN)
            self.xstep_itstat = None
        elif self.opt['PAR_CBPDN']:
            popt = parcbpdn.ParConvBPDN.Options(dict(self.opt['CBPDN']))
            xstep = parcbpdn.ParConvBPDN(self.D.squeeze(), S, lmbda, opt=popt,
                                         dimK=dimK, dimN=self.cri.dimN)
            xstep.solve()
            self.Sf = xstep.Sf
            self.setcoef(xstep.getcoef())
            self.xstep_itstat = xstep.itstat[-1] if len(xstep.itstat) > 0 \
                                                 else None
        else:
            # Create X update object (external representation is expected!)
            xstep = cbpdn.ConvBPDN(self.D.squeeze(), S, lmbda,
                                   self.opt['CBPDN'], dimK=dimK,
                                   dimN=self.cri.dimN)
            xstep.solve()
            self.Sf = xstep.Sf
            self.setcoef(xstep.getcoef())
            self.xstep_itstat = xstep.itstat[-1] if len(xstep.itstat) > 0 \
                                                 else None



    def setcoef(self, Z):
        """Set coefficient array."""

        # If the dictionary has a single channel but the input (and
        # therefore also the coefficient map array) has multiple
        # channels, the channel index and multiple image index have
        # the same behaviour in the dictionary update equation: the
        # simplest way to handle this is to just reshape so that the
        # channels also appear on the multiple image index.
        if self.cri.Cd == 1 and self.cri.C > 1:
            Z = Z.reshape(self.cri.Nv + (1,) + (self.cri.Cx*self.cri.K,) +
                          (self.cri.M,))
        self.Z[:] = np.asarray(Z, dtype=self.dtype)
        self.Zf = sl.rfftn(self.Z, self.cri.Nv, self.cri.axisN)



    def dstep(self):
        """Compute dictionary update for training data of preceding
        :meth:`xstep`.
        """

        # Compute X D - S
        Ryf = sl.inner(self.Zf, self.Df, axis=self.cri.axisM) - self.Sf
        # Compute gradient
        gradf = sl.inner(np.conj(self.Zf), Ryf, axis=self.cri.axisK)

        # If multiple channel signal, single channel dictionary
        if self.cri.C > 1 and self.cri.Cd == 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        # Update gradient step
        self.eta = self.eta_a / (self.j + self.eta_b)

        # Compute gradient descent
        self.Gf[:] = self.Df - self.eta * gradf
        self.G = sl.irfftn(self.Gf, self.cri.Nv, self.cri.axisN)

        # Eval proximal operator
        self.Dprv[:] = self.D
        self.D[:] = self.Pcn(self.G)



    def manage_itstat(self):
        """Compute, record, and display iteration statistics."""

        # Extract and record iteration stats
        itst = self.iteration_stats()
        self.itstat.append(itst)
        self.display_status(self.fmtstr, itst)



    def getdict(self):
        """Get final dictionary."""

        return self.D



    def itstat_extra(self):
        """Non-standard entries for the iteration stats record tuple."""

        return ()



    @classmethod
    def hdrtxt(cls):
        """Construct tuple of status display column title."""

        #  return ('Itn', 'X r', 'X s', u('X ρ'), 'D cnstr', 'D dlt', u('D η'), 'Time')
        return ('Itn', 'Fnc', 'DFid', 'l1', 'r_X', 's_X', u('ρ_X'),
                'Cnstr_D', 'dlt_D', u('η_D'), 'Time')



    @classmethod
    def hdrval(cls):
        """Construct dictionary mapping display column title to
        IterationStats entries.
        """

        #  hdrmap = {'Itn': 'Iter', 'X r': 'PrimalRsdl', 'X s': 'DualRsdl',
        #            u('X ρ'): 'Rho', 'D cnstr': 'Cnstr', 'D dlt': 'DeltaD',
        #            u('D η'): 'Eta', 'Time': 'Time'}
        hdrmap = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid', 'l1': 'RegL1',
                  'r_X': 'PrimalRsdl', 's_X': 'DualRsdl',
                  u('ρ_X'): 'Rho', 'Cnstr_D': 'Cnstr', 'dlt_D': 'DeltaD',
                  u('η_D'): 'Eta', 'Time': 'Time'}

        return hdrmap



    def iteration_stats(self):
        """Construct iteration stats record tuple."""

        tk = self.timer.elapsed(self.opt['IterTimer'])
        if self.xstep_itstat is None:
            objfn = (0.0,) * 3
            rsdl = (0.0,) * 2
            rho = (0.0,)
        else:
            objfn = (self.xstep_itstat.ObjFun, self.xstep_itstat.DFid,
                     self.xstep_itstat.RegL1)
            rsdl = (self.xstep_itstat.PrimalRsdl,
                    self.xstep_itstat.DualRsdl)
            rho = (self.xstep_itstat.Rho,)

        cnstr = linalg.norm(cr.zpad(self.D, self.cri.Nv) - self.G)
        dltd = linalg.norm(self.D - self.Dprv)

        tpl = (self.j,) + objfn + rsdl + rho + (cnstr, dltd, self.eta) + \
              self.itstat_extra() + (tk,)
        return type(self).IterationStats(*tpl)



    def getitstat(self):
        """Get iteration stats as named tuple of arrays instead of
        array of named tuples.
        """

        return util.transpose_ntpl_list(self.itstat)



    def display_config(self):
        """Set up status display if option selected. NB: this method
        assumes that the first entry is the iteration count and the
        last is the rho value.
        """

        if self.opt['Verbose']:
            hdrtxt = type(self).hdrtxt()
            # Call utility function to construct status display formatting
            self.hdrstr, self.fmtstr, self.nsep = common.solve_status_str(
                hdrtxt, fwdth0=type(self).fwiter, fprec=type(self).fpothr)
        else:
            self.hdrstr, self.fmtstr, self.nsep = '', '', 0



    def display_start(self):
        """Start status display if option selected."""

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print(self.hdrstr)
            print("-" * self.nsep)



    def display_status(self, fmtstr, itst):
        """Display current iteration status as selection of fields from
        iteration stats tuple.
        """

        if self.opt['Verbose']:
            hdrtxt = type(self).hdrtxt()
            hdrval = type(self).hdrval()
            itdsp = tuple([getattr(itst, hdrval[col]) for col in hdrtxt])

            print(fmtstr % itdsp)



    def display_end(self):
        """Terminate status display if option selected."""

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            print("-" * self.nsep)





class OnlineDictLearnSGDMask(OnlineDictLearnSGD):
    r"""
    Stochastic gradient descent (SGD) based online convolutional
    dictionary learning with a spatial mask, as proposed in
    :cite:`liu-2018-first`.
    """

    class Options(OnlineDictLearnSGD.Options):
        r"""Online masked CBPDN dictionary learning algorithm options.

        Options are the same as those of
        :class:`OnlineDictLearnSGD.Options`, except for

          ``CBPDN`` : Options :class:`.admm.cbpdn.ConvBPDNMaskDcpl.Options`.
        """

        defaults = copy.deepcopy(OnlineDictLearnSGD.Options.defaults)
        defaults.update({'CBPDN': copy.deepcopy(
                         cbpdn.ConvBPDNMaskDcpl.Options.defaults)})


        def __init__(self, opt=None):
            """Initialise online masked CBPDN dictionary learning
            algorithm options.
            """

            OnlineDictLearnSGD.Options.__init__(self, {
                'CBPDN': cbpdn.ConvBPDNMaskDcpl.Options({
                    'AutoRho': {'Period': 10, 'AutoScaling': False,
                    'RsdlRatio': 10.0, 'Scaling': 2.0, 'RsdlTarget': 1.0}})
                })

            if opt is None:
                opt = {}
            self.update(opt)



    def solve(self, S, W=None, dimK=None):
        """Compute sparse coding and dictionary update for training
        data `S`."""

        # Use dimK specified in __init__ as default
        if dimK is None and self.dimK is not None:
            dimK = self.dimK

        # Start solve timer
        self.timer.start(['solve', 'solve_wo_eval'])

        # Solve CSC problem on S and do dictionary step
        self.init_vars(S, dimK)
        if W is None:
            W = np.array([1.0], dtype=self.dtype)
        W = np.asarray(W.reshape(cr.mskWshape(W, self.cri)),
                       dtype=self.dtype)
        self.xstep(S, W, self.lmbda, dimK)
        self.dstep(W)

        # Stop solve timer
        self.timer.stop('solve_wo_eval')

        # Extract and record iteration stats
        self.manage_itstat()

        # Increment iteration count
        self.j += 1

        # Stop solve timer
        self.timer.stop('solve')

        # Return current dictionary
        return self.getdict()



    def xstep(self, S, W, lmbda, dimK):
        """Solve CSC problem for training data `S`."""

        if self.opt['CUDA_CBPDN']:
            Z = cucbpdn.cbpdnmsk(self.D.squeeze(), S[..., 0], W.squeeze(),
                                 lmbda, self.opt['CBPDN'])
            Z = Z.reshape(self.cri.Nv + (1, 1, self.cri.M,))
            self.Z[:] = np.asarray(Z, dtype=self.dtype)
            self.Zf = sl.rfftn(self.Z, self.cri.Nv, self.cri.axisN)
            self.Sf = sl.rfftn(S.reshape(self.cri.shpS), self.cri.Nv,
                               self.cri.axisN)
            self.xstep_itstat = None
        else:
            # Create X update object (external representation is expected!)
            xstep = cbpdn.ConvBPDNMaskDcpl(self.D.squeeze(), S, lmbda, W,
                                           self.opt['CBPDN'], dimK=dimK,
                                           dimN=self.cri.dimN)
            xstep.solve()
            self.Sf = sl.rfftn(S.reshape(self.cri.shpS), self.cri.Nv,
                               self.cri.axisN)
            self.setcoef(xstep.getcoef())
            self.xstep_itstat = xstep.itstat[-1] if xstep.itstat else None



    def dstep(self, W):
        """Compute dictionary update for training data of preceding
        :meth:`xstep`.
        """

        # Compute residual X D - S in frequency domain
        Ryf = sl.inner(self.Zf, self.Df, axis=self.cri.axisM) - self.Sf
        # Transform to spatial domain, apply mask, and transform back to
        # frequency domain
        Ryf[:] = sl.rfftn(W * sl.irfftn(Ryf, self.cri.Nv, self.cri.axisN),
                          None, self.cri.axisN)
        # Compute gradient
        gradf = sl.inner(np.conj(self.Zf), Ryf, axis=self.cri.axisK)

        # If multiple channel signal, single channel dictionary
        if self.cri.C > 1 and self.cri.Cd == 1:
            gradf = np.sum(gradf, axis=self.cri.axisC, keepdims=True)

        # Update gradient step
        self.eta = self.eta_a / (self.j + self.eta_b)

        # Compute gradient descent
        self.Gf[:] = self.Df - self.eta * gradf
        self.G = sl.irfftn(self.Gf, self.cri.Nv, self.cri.axisN)

        # Eval proximal operator
        self.Dprv[:] = self.D
        self.D[:] = self.Pcn(self.G)

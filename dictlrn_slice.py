# -*- coding: utf-8 -*-
"""Dictionary learning for slice-based convolutional sparse coding."""
import copy
import logging
import numpy as np
from scipy import linalg
from sporco.admm import admm
from sporco.dictlrn import dictlrn
from sporco.admm.cmod import getPcn
import sporco.linalg as sl
from sporco.util import u
from admm_slice import *

logger = logging.getLogger(__name__)


class ConvCnstrMODSliceBase(admm.ADMMEqual):
    r"""
    General base class for solving the problem

    .. math::
        \mathrm{argmin}_{D} \frac{1}{2}\sum_{i=1}^N \|D x_i - h_i\|_2^2
        \quad \text{such that} \quad D \in C

    where :math:`C` is the feasible set for the dictionary. This problem is
    solved via the ADMM formulation

    .. math::
        \mathrm{argmin}_{D} \frac{1}{2}\sum_{i=1}^N \|D x_i - h_i\|_2^2 +
        \iota_C(G) \quad \text{such that} \quad D = G.

    where :math:`\iota_C(\cdot)` is the indicator function for set :math:`C`.
    """

    class Options(admm.ADMMEqual.Options):

        defaults = copy.deepcopy(admm.ADMMEqual.Options.defaults)
        defaults.update({
            'AuxVarObj': True,
            'fEvalX': False,
            'gEvalY': True,
            'ReturnX': False,
            'RelaxParam': 1.8,
            'ZeroMean': False
        })
        defaults['AutoRho'].update({'Enabled': True})

        def __init__(self, opt=None):
            if opt is None:
                opt = {}
            super().__init__(opt)
            if self['AutoRho', 'RsdlTarget'] is None:
                self['AutoRho', 'RsdlTarget'] = 1.0

    # same as cmod.CnstrMOD
    itstat_fields_objfn = ('DFid', 'Cnstr')
    hdrtxt_objfn = ('DFid', 'Cnstr')
    hdrval_objfun = {'DFid': 'DFid', 'Cnstr': 'Cnstr'}

    def __init__(self, dinit, opt=None):
        """
        Initilize a ConvCnstrMODSliceBase object.

        Parameters
        ----------
        dinit: ndarray
            Initial value of the dictionary. The input dictionary should have
            compatible size of [patch_h, patch_w, channels, num_atoms].
        opt: :class:`ConvCnstrMODSliceBase.Options` object
            Options for our algorithm.
        """
        if opt is None:
            opt = ConvCnstrMODSliceBase.Options()
        # In our configuration the channel dimension comes before patch dim,
        # so transpose and reshape the dictionary to 2D.
        if dinit.ndim > 3:
            assert dinit.shape[-2] == 1 or dinit.shape[-2] == 3
            dinit = dinit.transpose(2, 0, 1, 3)
        else:
            # single channel dictionary. Insert channel dimension (1).
            dinit = np.expand_dims(dinit, axis=-2)
        self.dsz = dinit.shape
        dinit = dinit.reshape(-1, dinit.shape[-1])
        super().__init__(dinit.shape, dinit.dtype, opt)
        self.Y = dinit.copy()
        self.Pcn = getPcn(opt['ZeroMean'])
        # signals and coefficients array
        self.signals = None
        self.coefs = None

    def ystep(self):
        self.Y = self.Pcn(self.AX + self.U)

    def getdict(self):
        return self.getmin()

    def setcoef(self, coef_tuple):
        self.signals = np.asarray(coef_tuple[0], dtype=self.dtype)
        self.signals = self.signals.transpose((1, 0, 2))
        self.signals = self.signals.reshape(self.signals.shape[0], -1)
        self.coefs = np.asarray(coef_tuple[1], dtype=self.dtype)
        self.coefs = self.coefs.transpose((1, 0, 2))
        self.coefs = self.coefs.reshape(self.coefs.shape[0], -1)

    def eval_objfn(self):
        dfd = self.obfn_dfd()
        cns = self.obfn_cns()
        return (dfd, cns)

    def obfn_dfd(self):
        return 0.5*linalg.norm(
            np.matmul(self.obfn_fvar(), self.coefs) - self.signals
        )**2

    def obfn_cns(self):
        return linalg.norm(self.Pcn(self.obfn_gvar()) - self.obfn_gvar())

    def xstep(self):
        """To be implemented."""
        raise NotImplementedError


class ConvCnstrMODSliceMOD(ConvCnstrMODSliceBase):
    """MOD solver for slice-based MOD method."""

    class Options(ConvCnstrMODSliceBase.Options):
        defaults = copy.deepcopy(ConvCnstrMODSliceBase.Options.defaults)

        def __init__(self, opt=None):
            if opt is None:
                opt = {}
            super().__init__(opt)

    def __init__(self, dinit, opt=None):
        if opt is None:
            opt = ConvCnstrMODSliceMOD.Options()
        super().__init__(dinit, opt)

    def setcoef(self, coef_tuple):
        super().setcoef(coef_tuple)
        self.SZT = self.signals.dot(self.coefs.T)
        self.lu, self.piv = sl.lu_factor(self.coefs, self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)

    def xstep(self):
        self.X = np.asarray(
            sl.lu_solve_AATI(self.coefs, self.rho,
                             self.SZT+self.rho*(self.Y-self.U),
                             self.lu, self.piv),
            dtype=self.dtype
        )

    def rhochange(self):
        self.lu, self.piv = sl.lu_factor(self.coefs, self.rho)
        self.lu = np.asarray(self.lu, dtype=self.dtype)


class ConvBPDNSliceDictLearn(dictlrn.DictLearn):

    class Options(dictlrn.DictLearn.Options):
        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update({
            'AccurateDFid': False
        })

        def __init__(self, opt=None, xmethod=None, dmethod=None):
            if xmethod is None:
                xmethod = 'ConvBPDNSliceTwoBlockCnstrnt'
            if dmethod is None:
                dmethod = 'ConvCnstrMODSliceMOD'
            self.xmethod = xmethod
            self.dmethod = dmethod
            self.defaults.update({
                'CBPDN': globals()[xmethod].Options.defaults,
                'CCMOD': globals()[dmethod].Options.defaults
            })
            self.defaults['CBPDN'].update({'MaxMainIter': 1})
            self.defaults['CBPDN']['AutoRho'].update({
                'Period': 10, 'AutoScaling': False, 'RsdlRatio': 10.,
                'RsdlTarget': 1., 'Scaling': 2.
            })
            self.defaults['CCMOD'].update({'MaxMainIter': 1})
            self.defaults['CCMOD']['AutoRho'].update({
                'Period': 10, 'AutoScaling': False, 'RsdlRatio': 10.,
                'RsdlTarget': 1., 'Scaling': 2.
            })

            super().__init__({
                'CBPDN': globals()[xmethod].Options(self.defaults['CBPDN']),
                'CCMOD': globals()[dmethod].Options(self.defaults['CCMOD'])
            })
            if opt is None:
                opt = {}
            self.update(opt)

    def __init__(self, D0, S, lmbda=None, opt=None, xmethod=None,
                 dmethod=None, dimK=None, dimN=2):
        if opt is None:
            opt = ConvBPDNSliceDictLearn.Options(xmethod=xmethod,
                                                 dmethod=dmethod)
        self.opt = opt
        self.xmethod = opt.xmethod
        self.dmethod = opt.dmethod
        xstep = globals()[self.xmethod](D0, S, lmbda, opt['CBPDN'], dimK=dimK,
                                        dimN=dimN)
        dstep = globals()[self.dmethod](D0, opt['CCMOD'])
        isc = self.config_itstats()

        super().__init__(xstep, dstep, opt, isc)

    def config_itstats(self):
        """Config itstats output."""
        isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr', 'XPrRsdl',
                 'XDlRsdl', 'XRho', 'DPrRsdl', 'DDlRsdl', 'DRho', 'Time']
        isxmap = {
            'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl', 'XRho': 'Rho'
        }
        isdmap = {
            'DPrRsdl': 'PrimalRsdl', 'DDlRsdl': 'DualRsdl', 'DRho': 'Rho',
            'Cnstr': 'Cnstr'
        }
        hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr', 'r_X', 's_X',
                  u('ρ_X'), 'r_D', 's_D', u('ρ_D')]
        hdrmap = {
            'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
            u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr', 'r_X': 'XPrRsdl',
            's_X': 'XDlRsdl', u('ρ_X'): 'XRho', 'r_D': 'DPrRsdl',
            's_D': 'DDlRsdl', u('ρ_D'): 'DRho'
        }
        _evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        if self.opt['AccurateDFid']:
            evlmap = _evlmap
        else:
            evlmap = {}
            isxmap.update(_evlmap)
        return dictlrn.IterStatsConfig(
            isfld=isfld,
            isxmap=isxmap,
            isdmap=isdmap,
            evlmap=evlmap,
            hdrtxt=hdrtxt,
            hdrmap=hdrmap
        )

    def getcoef(self):
        """Get final coefficient map with standard layout."""
        return self.xstep.getmin()

    def getdict(self):
        """Get final dictionary with standard layout.

        Returns
        -------
        D: array, [patch_h, patch_w, channels, num_atoms]
        """
        D = self.dstep.getdict().reshape(self.dstep.dsz)
        # transpose the channel dimension back
        D = D.transpose(1, 2, 0, 3)
        return D

    def reconstruct(self, D=None, X=None):
        """Reconstruct representation."""
        if D is None:
            D = self.getdict()
        if X is None:
            X = self.getcoef()
        # reshape to 2D representation
        D = D.transpose(2, 0, 1, 3)
        D = D.reshape(-1, D.shape[-1])
        X = X.transpose(3, 4, 0, 1, 2)
        X = X.reshape(X.shape[0], X.shape[1], -1)
        recon = self.xstep.slices2im(np.matmul(D, X))
        recon = np.expand_dims(recon.transpose(2, 3, 1, 0), axis=-1)
        return recon

    def evaluate(self):
        """Evaluate functional value."""
        if self.opt['AccurateDFid']:
            D = self.dstep.getmin()
            X = self.xstep.getmin()
            recon = self.xstep.slices2im(D.dot(X))
            dfd = ((recon - self.xstep.S) ** 2).sum() / 2.0
            reg = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=reg, ObjFun=dfd+self.xstep.lmbda*reg)
        return None

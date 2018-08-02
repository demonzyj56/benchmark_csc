# -*- coding: utf-8 -*-
"""Online version of slice based CDL with FISTA solver."""
import copy
import collections
import logging
from future.utils import with_metaclass
import numpy as np
from scipy import linalg
import sporco.cnvrep as cr
import sporco.util as su
import sporco.linalg as sl
from sporco.util import u
from sporco import common, cdict
from sporco.dictlrn import dictlrn
from sporco.admm import cbpdn
from sporco.fista import fista
import torch

from dictlrn_slice import Pcn
from im2slices import im2slices, slices2im

logger = logging.getLogger(__name__)


def einsum(subscripts, operands):
    """Wrapper around possible implementations of einsum."""
    if 0:
        out = np.einsum(subscripts, *operands)
    else:
        operands = [torch.from_numpy(o) for o in operands]
        out = torch.einsum(subscripts, operands).numpy()
    return out


class IterStatsConfig(object):

    fwiter = 4
    """Field width for iteration count display column"""
    fpothr = 2
    """Field precision for other display columns"""

    def __init__(self, isfld, isxmap, isdmap, evlmap, hdrtxt, hdrmap,
                 fmtmap=None):
        self.IterationStats = collections.namedtuple('IterationStats', isfld)
        self.isxmap = isxmap
        self.isdmap = isdmap
        self.evlmap = evlmap
        self.hdrtxt = hdrtxt
        self.hdrmap = hdrmap

        # Call utility function to construct status display formatting
        self.hdrstr, self.fmtstr, self.nsep = common.solve_status_str(
            hdrtxt, fmtmap=fmtmap, fwdth0=type(self).fwiter,
            fprec=type(self).fpothr)

    def iterstats(self, j, t, dtx, dtd, evl):
        """Construct IterationStats namedtuple from X step and D step
        statistics.

        Parameters
        ----------
        j: int
            Iteration number
        t: float
            Iteration time
        dtx: dict
            Dict holding statistics from X step
        dtd: dict
            Dict holding statistics from D step
        evl: dict
            Dict holding statistics from extra evaluations
        """
        vlst = []
        for fnm in self.IterationStats._fields:
            if fnm in self.isxmap:
                vlst.append(dtx[self.isxmap[fnm]])
            elif fnm in self.isdmap:
                vlst.append(dtd[self.isdmap[fnm]])
            elif fnm in self.evlmap:
                vlst.append(evl[fnm])
            elif fnm == 'Iter':
                vlst.append(j)
            elif fnm == 'Time':
                vlst.append(t)
            else:
                vlst.append(None)

        return self.IterationStats._make(vlst)

    def printheader(self):
        self.print_func(self.hdrstr)
        self.printseparator()

    def printseparator(self):
        self.print_func("-" * self.nsep)

    def printiterstats(self, itst):
        """Print iteration statistics.

        Parameters
        ----------
        itst : namedtuple
            IterationStats namedtuple as returned by :meth:`iterstats`
        """

        itdsp = tuple([getattr(itst, self.hdrmap[col]) for col in self.hdrtxt])
        self.print_func(self.fmtstr % itdsp)

    def print_func(self, s):
        """Print function."""
        print(s)


class StripeSliceFISTA(fista.FISTA):
    r"""FISTA algorithm to solve for the dictionary, where the derivative is
    given by

    .. math::
        \nabla_D f = \Omega At - Bt,

    where :math:`\Omega` is the stripe dictionary.
    """

    class Options(fista.FISTA.Options):
        defaults = copy.deepcopy(fista.FISTA.Options.defaults)
        defaults.update({
            'ZeroMean': True
        })

        def __init__(self, opt=None):
            if opt is None:
                opt = {}
            super().__init__(opt)

    itstat_fields_objfn = ('Cnstr', )
    hdrtxt_objfn = ('Cnstr', )
    hdrval_objfun = {'Cnstr': 'Cnstr'}

    def __init__(self, At, Bt, dsz=None, opt=None):
        if opt is None:
            opt = StripeSliceFISTA.Options()
        if opt['X0'] is not None:
            self.dsz = opt['X0'].shape
        else:
            self.dsz = dsz
        super().__init__(np.prod(dsz), dsz, At.dtype, opt)
        self.At = At
        self.Bt = Bt
        self.Y = self.X.copy()
        self.osz = list(copy.deepcopy(self.dsz))
        self.osz[2] = 2 * self.osz[0] - 1
        self.osz[3] = 2 * self.osz[1] - 1
        self.Omega = np.zeros(self.osz, dtype=self.dtype)

    def set_Omega(self, D=None):
        r"""Set the stripe dictionary :math:`\Omega` from D."""
        if D is None:
            D = self.Y
        self.Omega.fill(0.)
        for ih, h in enumerate(range(-self.osz[0]+1, self.osz[0])):
            for iw, w in enumerate(range(-self.osz[1]+1, self.osz[1])):
                begh = -min(h, 0)
                endh = self.osz[0] - max(h, 0)
                begw = -min(w, 0)
                endw = self.osz[1] - max(w, 0)
                stripe_dict = D[begh:endh, begw:endw, 0, 0, ...]
                begh_c = max(h, 0)
                endh_c = min(self.osz[0] + h, self.osz[0])
                begw_c = max(w, 0)
                endw_c = min(self.osz[1] + w, self.osz[1])
                self.Omega[begh_c:endh_c, begw_c:endw_c, ih, iw, ...] = \
                    stripe_dict

    def Pcn(self, D):
        """Proximal function."""
        return Pcn(D, self.opt['ZeroMean'])

    def eval_grad(self):
        r"""Evaluate the gradient:

            .. ::math:
                \nabla_D f = \Omega At - Bt
        """
        self.set_Omega()
        grad = einsum('ijklmno,klpqros->ijpqmns', (self.Omega, self.At))
        grad -= self.Bt
        return grad

    def eval_proxop(self, V):
        return self.Pcn(V)

    def rsdl(self):
        """Fixed point residual."""
        return linalg.norm(self.X - self.Yprv)

    def eval_objfn(self):
        """Eval constraint only."""
        cnstr = linalg.norm(self.X - self.Pcn(self.X))
        return (cnstr, )


class OnlineSliceDictLearn2nd(with_metaclass(dictlrn._DictLearn_Meta,
                                             common.BasicIterativeSolver)):
    r"""Stochastic Approximation based online convolutional dictionary
    learning.
    """

    class Options(cdict.ConstrainedDict):
        defaults = {
            'Verbose': True, 'StatusHeader': True, 'IterTimer': 'solve',
            'MaxMainIter': 1000, 'Callback': None,
            'AccurateDFid': False, 'Boundary': 'circulant_back',
            'BatchSize': 32, 'DataType': None, 'StepIter': -1,
            'CBPDN': copy.deepcopy(cbpdn.ConvBPDN.Options.defaults),
            'CCMOD': copy.deepcopy(StripeSliceFISTA.Options.defaults),
            'OCDL': {
                'p': 1.,  # forgetting exponent

            }
        }
        defaults['CBPDN'].update({
            'Verbose': False, 'MaxMainIter': 50, 'AuxVarObj': False,
            'RelStopTol': 1e-7, 'DataType': None, 'FastSolve': True,
        })
        defaults['CBPDN']['AutoRho'].update({'Enabled': False})
        defaults['CCMOD']['BackTrack'].update({'Enabled': False})

        def __init__(self, opt=None):
            super().__init__({
                'CBPDN': cbpdn.ConvBPDN.Options(self.defaults['CBPDN']),
                'CCMOD': StripeSliceFISTA.Options(self.defaults['CCMOD']),
            })
            if opt is None:
                opt = {}
            self.update(opt)

    def __new__(cls, *args, **kwargs):
        instance = super(OnlineSliceDictLearn2nd, cls).__new__(cls)
        instance.timer = su.Timer(['init', 'solve', 'solve_wo_eval',
                                   'xstep', 'dstep'])
        instance.timer.start('init')
        return instance

    def __init__(self, D0, S0, lmbda=None, opt=None, dimK=1, dimN=2):
        """Internally we use a 7-dim representation over blobs. This increases
        the spatial dimension of 2 to 4 to allow for extra dimensions for
        slices.

        -------------------------------------------------------------------
        blob     | spatial                                ,chn  ,sig  ,fil
        -------------------------------------------------------------------
        S        |  (H      ,  W      ,  1      ,  1      ,  C  ,  K  ,  1)
        D        |  (Hc     ,  Wc     ,  1      ,  1      ,  C  ,  1  ,  M)
        X        |  (H      ,  W      ,  1      ,  1      ,  1  ,  K  ,  M)
        Omega    |  (Hc     ,  Wc     ,  2Hc-1  ,  2Wc-1  ,  C  ,  1  ,  M)
        At       |  (2Hc-1  ,  2Wc-1  ,  1      ,  1      ,  1  ,  M  ,  M)
        Bt       |  (Hc     ,  Wc     ,  1      ,  1      ,  C  ,  1  ,  M)
        patches  |  (H      ,  W      ,  Hc     ,  Wc     ,  C  ,  K  ,  1)
        gamma    |  (H      ,  W      ,  2Hc-1  ,  2Wc-1  ,  1  ,  K  ,  M)
        -------------------------------------------------------------------

        Here the `signal` dimension of At is occupied by M, which comes from
        stripe dictionary Omega.
        """
        if opt is None:
            opt = OnlineSliceDictLearn2nd.Options()
        assert isinstance(opt, OnlineSliceDictLearn2nd.Options)
        self.opt = opt

        self.set_dtype(opt, S0.dtype)

        # insert extra dims
        D0 = D0[:, :, np.newaxis, np.newaxis, ...]
        S0 = S0[:, :, np.newaxis, np.newaxis, ...]

        assert dimN == 2
        self.cri = cr.CSC_ConvRepIndexing(D0, S0, dimK=None, dimN=4)
        self.osz = list(copy.deepcopy(self.cri.shpD))
        self.osz[2], self.osz[3] = 2*self.osz[0]-1, 2*self.osz[1]-1

        self.isc = self.config_itstats()
        self.itstat = []
        self.j = 0

        self.set_attr('lmbda', lmbda, dtype=self.dtype)
        self.set_attr('boundary', opt['Boundary'], dval='circulant_back')

        D0 = Pcn(D0, opt['CCMOD', 'ZeroMean'])

        self.D = np.asarray(D0.reshape(self.cri.shpD), dtype=self.dtype)
        self.S0 = np.asarray(S0.reshape(self.cri.shpS), dtype=self.dtype)
        self.At = self.dtype.type(0.)
        self.Bt = self.dtype.type(0.)

        self.lmbda = self.dtype.type(lmbda)
        self.Lmbda = self.dtype.type(0.)
        self.p = self.dtype.type(self.opt['OCDL', 'p'])

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            self.isc.printheader()

    def solve(self, S):
        """Solve for given signal S."""
        self.cri = cr.CSC_ConvRepIndexing(
            self.D.squeeze()[:, :, None, None, ...],
            S[:, :, None, None, ...],
            dimK=None, dimN=4
        )

        self.timer.start(['solve', 'solve_wo_eval'])

        # Initialize with CBPDN
        self.timer.start('xstep')
        xstep = cbpdn.ConvBPDN(self.getdict(), S, self.lmbda,
                               opt=self.opt['CBPDN'])
        xstep.solve()
        self.timer.stop('xstep')
        X = np.asarray(xstep.getcoef().reshape(self.cri.shpX), dtype=self.dtype)

        # update At and Bt
        patches = self.im2slices(S)
        self.update_At(X)
        self.update_Bt(X, patches)
        self.Lmbda = self.dtype.type(self.alpha*self.Lmbda+1)

        # update dictionary with FISTA
        fopt = copy.deepcopy(self.opt['CCMOD'])
        fopt['X0'] = self.D
        self.timer.start('dstep')
        dstep = StripeSliceFISTA(self.At, self.Bt, opt=fopt)
        dstep.solve()
        self.timer.stop('dstep')

        # set dictionary
        self.setdict(dstep.getmin())

        self.timer.stop('solve_wo_eval')
        evl = self.evaluate(S.reshape(self.cri.shpS), X)
        self.timer.start('solve_wo_eval')

        t = self.timer.elapsed(self.opt['IterTimer'])
        itst = self.isc.iterstats(self.j, t, xstep.itstat[-1], dstep.itstat[-1],
                                  evl)
        self.itstat.append(itst)

        if self.opt['Verbose']:
            self.isc.printiterstats(itst)

        self.j += 1

        self.timer.stop(['solve', 'solve_wo_eval'])

        return self.getdict()

    def config_itstats(self):
        """Setup config fields."""
        # NOTE: BackTrack is not implemented so always False.
        isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1',
                 'XPrRsdl', 'XDlRsdl', 'XRho', 'X_It',
                 'D_L', 'D_Rsdl', 'D_It', 'Time']
        isxmap = {'XPrRsdl': 'PrimalRsdl', 'XDlRsdl': 'DualRsdl',
                  'XRho': 'Rho', 'X_It': 'Iter'}
        isdmap = {'D_L': 'L', 'D_Rsdl': 'Rsdl', 'D_It': 'Iter'}
        hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'),
                  'Itn_X', 'r_X', 's_X', u('ρ_X'),
                  'Itn_D', 'r_D', 'L_D']
        hdrmap = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                  u('ℓ1'): 'RegL1', 'r_X': 'XPrRsdl', 's_X': 'XDlRsdl',
                  u('ρ_X'): 'XRho', 'Itn_X': 'X_It',
                  'r_D': 'D_Rsdl', 'L_D': 'D_L', 'Itn_D': 'D_It'}
        if self.opt['AccurateDFid']:
            evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        else:
            evlmap = {}
        return dictlrn.IterStatsConfig(
            isfld=isfld, isxmap=isxmap, isdmap=isdmap, evlmap=evlmap,
            hdrtxt=hdrtxt, hdrmap=hdrmap,
            fmtmap={'Itn_X': '%4d', 'Itn_D': '%4d'}
        )

    def getdict(self):
        """getdict() returns a squeezed version of internal dictionary."""
        return self.D.squeeze()

    def setdict(self, D=None):
        """Set dictionary properly."""
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)

    def update_At(self, X):
        r"""Update At. At is computed as following:

        .. math::
            At = \sum \gamma_i x_i^T.

        """
        gamma = self.stripe_slice(X)
        Lmbda_new = self.dtype.type(self.alpha*self.Lmbda+1)
        new = einsum('ijklmno,ijpqrns->klpqmos', (gamma, X))
        self.At = np.asarray(
            (self.At*self.alpha*self.Lmbda+new) * (1./Lmbda_new),
            dtype=self.dtype
        )

    def update_Bt(self, X, patches):
        r"""Update Bt. Bt is computed as following:

        .. math::
            Bt = \sum s_i x_i^T.

        """
        Lmbda_new = self.dtype.type(self.alpha*self.Lmbda+1)
        new = einsum('ijklmno,ijpqrns->klpqmos', (patches, X))
        self.Bt = np.asarray(
            (self.Bt*self.alpha*self.Lmbda+new) * (1./Lmbda_new),
            dtype=self.dtype
        )

    def stripe_slice(self, X):
        r"""Construct stripe slice (:math:`\gamma`) from sparse code X."""
        Hc, Wc = self.cri.shpD[:2]
        sz = list(copy.deepcopy(X.shape))
        sz[2], sz[3] = 2*Hc-1, 2*Wc-1
        slices = np.zeros(sz, dtype=X.dtype)
        pad = [(Hc-1, Hc-1), (Wc-1, Wc-1)] + [(0, 0) for _ in range(X.ndim-2)]
        Xp = np.pad(X, pad, 'wrap')
        for h in range(X.shape[0]):
            for w in range(X.shape[1]):
                gamma = Xp[h:h+sz[2], w:w+sz[3], 0, 0, ...]  # 5D
                slices[h, w, ...] = gamma
        return slices

    def im2slices(self, S):
        """Convert signals to patches."""
        kernel_h, kernel_w = self.cri.shpD[:2]
        if self.cri.C == 1:
            S = S.squeeze().transpose(2, 0, 1)[:, np.newaxis, :, :]
        else:
            assert S.shape[-2] == self.cri.C
            S = S.transpose(3, 2, 0, 1)
        # [K, C*Hc*Wc, H*W]
        slices = im2slices(S, kernel_h, kernel_w, self.boundary)
        shp = [slices.shape[0], self.cri.C, kernel_h, kernel_w,
               S.shape[-2], S.shape[-1], 1]
        # [K, C, Hc, Wc, H, W, 1] -> [H, W, Hc, Wc, C, K, 1]
        slices = slices.reshape(shp)
        slices = slices.transpose(4, 5, 2, 3, 1, 0, 6)
        return slices

    @property
    def alpha(self):
        """Forgetting factor."""
        # j starts from 0
        alpha = self.dtype.type(pow(1.-1./(self.j+1.), self.p))
        return alpha

    def evaluate(self, S, X):
        """Optionally evaluate functional values."""
        if self.opt['AccurateDFid']:
            Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
            Xf = sl.rfftn(X, self.cri.Nv, self.cri.axisN)
            Sf = sl.rfftn(S, self.cri.Nv, self.cri.axisN)
            Ef = sl.inner(Df, Xf, axis=self.cri.axisM) - Sf
            dfd = sl.rfl2norm2(Ef, S.shape, axis=self.cri.axisN) / 2.
            rl1 = np.sum(np.abs(X))
            evl = dict(DFid=dfd, RegL1=rl1, ObjFun=dfd+self.lmbda*rl1)
        else:
            evl = None
        return evl


class OnlineSliceDictLearn(with_metaclass(dictlrn._DictLearn_Meta,
                                          common.BasicIterativeSolver)):

    class Options(cdict.ConstrainedDict):
        defaults = {
            'Verbose': True, 'StatusHeader': True, 'IterTimer': 'solve',
            'MaxMainIter': 1000, 'Callback': None,
            'AccurateDFid': False, 'Boundary': 'circulant_back',
            'BatchSize': 32, 'DataType': None, 'StepIter': -1,
            'CBPDN': {
                'Verbose': False, 'MaxMainIter': 50,
                'AutoRho': {'Enabled': False}, 'AuxVarObj': False,
                'RelStopTol': 1e-7, 'DataType': None,
                'FastSolve': True
            },
            'FISTA': {
                'L': 1., 'BackTrack': {'Enabled': False},
            },
            'CCMOD': {
                'L': 1., 'BackTrack': {'Enabled': False},
                'ZeroMean': False,
            },

        }

        def __init__(self, opt=None):
            super().__init__(
                {'CBPDN': cbpdn.ConvBPDN.Options(self.defaults['CBPDN'])}
            )
            if opt is None:
                opt = {}
            self.update(opt)

    def __new__(cls, *args, **kwargs):
        instance = super(OnlineSliceDictLearn, cls).__new__(cls)
        instance.timer = su.Timer(['init', 'solve', 'solve_wo_eval',
                                   'xstep', 'dstep'])
        instance.timer.start('init')
        return instance

    def __init__(self, D0, S0, lmbda=None, opt=None, dimK=1, dimN=2):
        if opt is None:
            opt = OnlineSliceDictLearn.Options()
        self.opt = opt

        self.set_dtype(opt, S0.dtype)

        self.cri = cr.CSC_ConvRepIndexing(D0, S0, dimK, dimN)

        self.isc = self.config_itstats()

        self.itstat = []
        self.j = 0

        # config for real application
        self.set_attr('lmbda', lmbda, dtype=self.dtype)
        self.set_attr('boundary', opt['Boundary'], dval='circulant_back')

        self.Sval = np.asarray(S0.reshape(self.cri.shpS), dtype=self.dtype)
        # [N, C, H, W]
        self.Sval = self.Sval.squeeze(-1).transpose(3, 2, 0, 1)
        self.Sval_slice = self.im2slices(self.Sval)

        self.dsz = D0.shape
        D0 = Pcn(D0, opt['CCMOD', 'ZeroMean'])
        self.D = np.asarray(D0.reshape(self.cri.shpD), dtype=self.dtype)
        # [Cin, Hc, Wc, Cout]; channel first
        self.D = self.D.squeeze(-2).transpose(2, 0, 1, 3)
        self.D = self.D.reshape(-1, self.D.shape[-1])

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            self.isc.printheader()

    def xinit(self, S):
        init = cbpdn.ConvBPDN(self.getdict().squeeze(), S, self.lmbda,
                              self.opt['CBPDN'])
        init.solve()
        # (H, W, 1, K, M)
        X = init.getcoef()
        assert X.shape[2] == 1
        X = X.reshape(-1, X.shape[-2], X.shape[-1])
        X = X.transpose(1, 2, 0)
        return X

    def solve(self, S):
        """Solve for given signal S."""

        self.timer.start(['solve', 'solve_wo_eval'])

        cri = cr.CDU_ConvRepIndexing(self.dsz, S)
        S = np.asarray(S.reshape(cri.shpS), dtype=self.dtype)

        X = self.xinit(S)
        Y = X.copy()
        G = self.D.copy()
        D = self.D.copy()
        S = S.squeeze(-1).transpose(3, 2, 0, 1)
        tx = td = 1.

        # MaxMainIter gives no of iterations for each sample.
        for self.j in range(self.j, self.j + self.opt['MaxMainIter']):

            Xprev, Dprev = X.copy(), D.copy()

            Y, X, G, D, tx, td = self.step(S, Y, X, G, D, tx, td,
                                           self.opt['FISTA', 'L'],
                                           self.opt['CCMOD', 'L'])
            tx = (1 + np.sqrt(1 + 4 * tx ** 2)) / 2.
            td = (1 + np.sqrt(1 + 4 * td ** 2)) / 2.

            self.timer.stop('solve_wo_eval')
            X_Rsdl = linalg.norm(Y - Xprev)
            D_Rsdl = linalg.norm(G - Dprev)
            recon = self.slices2im(np.matmul(G, Y))
            dfd = linalg.norm(recon - S) ** 2 / 2.
            reg = linalg.norm(Y.ravel(), 1)
            obj = dfd + self.lmbda * reg
            cnstr = linalg.norm(self.dprox(G) - G)
            dtx = {'L': self.opt['FISTA', 'L'], 'Rsdl': X_Rsdl,
                   'F_Btrack': None, 'Q_Btrack': None, 'IterBTrack': None}
            dtd = {'L': self.opt['CCMOD', 'L'], 'Rsdl': D_Rsdl, 'Cnstr': cnstr,
                   'F_Btrack': None, 'IterBTrack': None, 'Q_Btrack': None}
            evl = {'ObjFun': obj, 'DFid': dfd, 'RegL1': reg}
            if not self.opt['AccurateDFid']:
                dtx.update(evl)
                evl = None
            self.timer.start('solve_wo_eval')

            self.D = G
            self.X = Y

            t = self.timer.elapsed(self.opt['IterTimer'])
            itst = self.isc.iterstats(self.j, t, dtx, dtd, evl)

            if self.opt['Verbose']:
                self.isc.printiterstats(itst)

            if self.opt['Callback'] is not None:
                if self.opt['Callback'](self):
                    break

            if 0:
                import matplotlib.pyplot as plt
                plt.imshow(su.tiledict(self.getdict().squeeze()))
                plt.show()

        self.j += 1

        self.timer.stop(['solve', 'solve_wo_eval'])

        if self.opt['Verbose'] and self.opt['StatusHeader']:
            self.isc.printseparator()

        return self.getdict()

    def step(self, S, Y, X, G, D, tx, td, Lx, Ld):
        """Step for one iteration for given signal S."""
        recon = self.slices2im(np.matmul(G, Y))
        rsdl_slice = self.im2slices(recon - S)
        K = rsdl_slice.shape[0]

        def _rshp_3d_2d(blob):
            blob = blob.transpose(1, 0, 2)
            blob = blob.reshape(blob.shape[0], -1)
            return blob

        def _rshp_2d_3d(blob, K):
            blob = blob.reshape(blob.shape[0], K, -1)
            blob = blob.transpose(1, 0, 2)
            return blob

        # reshape from 3d blob (K, n/m, N) to (n/m, K*N)
        X = _rshp_3d_2d(X)
        Y = _rshp_3d_2d(Y)
        rsdl_slice = _rshp_3d_2d(rsdl_slice)

        idx = np.random.permutation(rsdl_slice.shape[-1])
        cur_idx = 0
        batch_size = min(self.opt['BatchSize'], len(idx))
        num_iter = self.opt['StepIter']
        if num_iter < 0:
            num_iter = len(idx) // batch_size
        for _ in range(num_iter):
            if cur_idx + batch_size > len(idx):
                idx = np.random.permutation(rsdl_slice.shape[-1])
                cur_idx = 0
            cur = idx[cur_idx:cur_idx+batch_size]
            Gnew, Dnew = self.dstep(G, D, Y[..., cur], rsdl_slice[..., cur], td, Ld)
            Ynew, Xnew = self.xstep(Y[..., cur], X[..., cur], Gnew,
                                    rsdl_slice[..., cur], tx, Lx)
            Y[..., cur] = Ynew
            X[..., cur] = Xnew
            G = Gnew
            D = Dnew
            cur_idx += batch_size

        # reshape back from 2d blob (n/m, K*N) to 3d (K, n/m, N)
        X = _rshp_2d_3d(X, K)
        Y = _rshp_2d_3d(Y, K)
        return Y, X, G, D, tx, td

    def xstep(self, Y, X, D, rsdl_slice, tx, Lx):
        """Do one step of FISTA on X."""
        self.timer.start('xstep')
        grad = np.matmul(D.T, rsdl_slice)
        Ynew = self.xprox(X - grad / Lx, Lx)
        tnew = (1 + np.sqrt(1 + 4 * tx ** 2)) / 2.
        Xnew = Ynew + (tx - 1) / tnew * (Ynew - Y)
        self.timer.stop('xstep')
        return Ynew, Xnew

    def dstep(self, G, D, X, rsdl_slice, td, Ld):
        """Do one step of FISTA on D."""
        self.timer.start('dstep')
        if rsdl_slice.ndim == 2:
            grad = np.matmul(rsdl_slice, X.T)
        else:
            grad = np.tensordot(rsdl_slice, X, axes=((0, 2), (0, 2)))
        Gnew = self.dprox(D - grad / Ld)
        tnew = (1 + np.sqrt(1 + 4 * td ** 2)) / 2.
        Dnew = Gnew + (td - 1) / tnew * (Gnew - G)
        self.timer.stop('dstep')
        return Gnew, Dnew

    def xprox(self, X, Lx):
        """Proximal operator for sparse representation."""
        return sl.shrink1(X, self.lmbda / Lx)

    def dprox(self, D):
        """Proximal operator for dictionary."""
        return Pcn(D, self.opt['CCMOD', 'ZeroMean'])

    def config_itstats(self):
        """Config itstats output for fista."""
        # isfld
        isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr']
        if self.opt['FISTA', 'BackTrack', 'Enabled']:
            isfld.extend(['X_F_Btrack', 'X_Q_Btrack', 'X_ItBt', 'X_L',
                          'X_Rsdl'])
        else:
            isfld.extend(['X_L', 'X_Rsdl'])
        if self.opt['CCMOD', 'BackTrack', 'Enabled']:
            isfld.extend(['D_F_Btrack', 'D_Q_Btrack', 'D_ItBt', 'D_L',
                          'D_Rsdl'])
        else:
            isfld.extend(['D_L', 'D_Rsdl'])
        isfld.extend(['Time'])
        # isxmap/isdmap
        isxmap = {'X_F_Btrack': 'F_Btrack', 'X_Q_Btrack': 'Q_Btrack',
                  'X_ItBt': 'IterBTrack', 'X_L': 'L', 'X_Rsdl': 'Rsdl'}
        isdmap = {'Cnstr': 'Cnstr', 'D_F_Btrack': 'F_Btrack',
                  'D_Q_Btrack': 'Q_Btrack', 'D_ItBt': 'IterBTrack',
                  'D_L': 'L', 'D_Rsdl': 'Rsdl'}
        # hdrtxt
        hdrtxt = ['Itn', 'Fnc', 'DFid', u('ℓ1'), 'Cnstr']
        if self.opt['FISTA', 'BackTrack', 'Enabled']:
            hdrtxt.extend(['F_X', 'Q_X', 'It_X', 'L_X'])
        else:
            hdrtxt.append('L_X')
        if self.opt['CCMOD', 'BackTrack', 'Enabled']:
            hdrtxt.extend(['F_D', 'Q_D', 'It_D', 'L_D'])
        else:
            hdrtxt.append('L_D')
        # hdrmap
        hdrmap = {'Itn': 'Iter', 'Fnc': 'ObjFun', 'DFid': 'DFid',
                  u('ℓ1'): 'RegL1', 'Cnstr': 'Cnstr'}
        if self.opt['FISTA', 'BackTrack', 'Enabled']:
            hdrmap.update({'F_X': 'X_F_Btrack', 'Q_X': 'X_Q_Btrack',
                           'It_X': 'X_ItBt', 'L_X': 'X_L'})
        else:
            hdrmap.update({'L_X': 'X_L'})
        if self.opt['CCMOD', 'BackTrack', 'Enabled']:
            hdrmap.update({'F_D': 'D_F_Btrack', 'Q_D': 'D_Q_Btrack',
                           'It_D': 'D_ItBt', 'L_D': 'D_L'})
        else:
            hdrmap.update({'L_D': 'D_L'})
        # evlmap
        _evlmap = {'ObjFun': 'ObjFun', 'DFid': 'DFid', 'RegL1': 'RegL1'}
        if self.opt['AccurateDFid']:
            evlmap = _evlmap
        else:
            evlmap = {}
            isxmap.update(_evlmap)
        # fmtmap
        fmtmap = {'It_X': '%4d', 'It_D': '%4d'}
        return IterStatsConfig(
            isfld=isfld,
            isxmap=isxmap,
            isdmap=isdmap,
            evlmap=evlmap,
            hdrtxt=hdrtxt,
            hdrmap=hdrmap,
            fmtmap=fmtmap
        )

    def im2slices(self, S):
        kernel_h, kernel_w = self.cri.shpD[:2]
        return im2slices(S, kernel_h, kernel_w, self.boundary)

    def slices2im(self, slices):
        kernel_h, kernel_w = self.cri.shpD[:2]
        output_h, output_w = self.cri.shpS[:2]
        return slices2im(slices, kernel_h, kernel_w, output_h, output_w,
                         self.boundary)

    def getdict(self):
        """Get final dictionary with standard layout.

        Returns
        -------
        D: array, [patch_h, patch_w, channels, num_atoms]
        """
        D = self.D.copy()
        patch_h, patch_w = self.cri.shpD[:2]
        D = D.reshape(-1, patch_h, patch_w, D.shape[-1])
        assert D.shape[0] == 1 or D.shape[0] == 3
        D = D.transpose(1, 2, 0, 3).squeeze()
        return D

    def reconstruct(self, D=None, X=None):
        if D is None:
            D = self.getdict()
        if D.ndim == 3:
            D = np.expand_dims(D, axis=-2)
        D = D.transpose(2, 0, 1, 3)
        D = D.reshape(-1, D.shape[-1])
        if X is None:
            X = self.X.copy()
        else:
            X = X.transpose(3, 4, 0, 1, 2)
            X = X.reshape(X.shape[0], X.shape[1], -1)
        recon = self.slices2im(np.matmul(D, X))
        recon = np.expand_dims(recon.transpose(2, 3, 1, 0), axis=-1)
        return recon

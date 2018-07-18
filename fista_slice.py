# -*- coding: utf-8 -*-
import copy
import logging
import numpy as np
from scipy import linalg
import sporco.cnvrep as cr
import sporco.linalg as sl
from sporco.util import u
from sporco.fista import fista
from sporco.dictlrn import dictlrn
from im2slices import im2slices, slices2im
from dictlrn_slice import Pcn

logger = logging.getLogger(__name__)


class ConvBPDNSliceFISTA(fista.FISTA):

    class Options(fista.FISTA.Options):
        defaults = copy.deepcopy(fista.FISTA.Options.defaults)
        defaults.update({
            'Boundary': 'circulant_back',
        })

        def __init__(self, opt=None):
            if opt is None:
                opt = {}
            super().__init__(opt)

    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}

    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
        if opt is None:
            opt = ConvBPDNSliceFISTA.Options()

        if not hasattr(self, 'cri'):
            self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        self.set_dtype(opt, S.dtype)
        self.lmbda = self.dtype.type(lmbda)
        # set boundary condition
        self.set_attr('boundary', opt['Boundary'], dval='circulant_back',
                      dtype=None)
        self.setdict(D)

        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=S.dtype)
        self.S = self.S.squeeze(-1).transpose((3, 2, 0, 1))
        self.S_slice = self.im2slices(self.S)
        xshape = (self.S_slice.shape[0], self.D.shape[1],
                  self.S_slice.shape[-1])
        Nx = np.prod(xshape)
        super().__init__(Nx, xshape, S.dtype, opt)
        if self.opt['BackTrack', 'Enabled']:
            self.L /= self.lmbda
        self.Y = self.X.copy()
        self.residual = -self.S_slice.copy()

    def eval_grad(self):
        return np.matmul(self.D.T, self.residual)

    def eval_proxop(self, V):
        return sl.shrink1(V, self.lmbda/self.L)

    def eval_R(self, V):
        recon = self.slices2im(np.matmul(self.D, V))
        return linalg.norm(self.S - recon) ** 2 / 2.

    def combination_step(self):
        super().combination_step()
        self.residual = self.im2slices(
            self.slices2im(np.matmul(self.D, self.Y))-self.S
        )

    def rsdl(self):
        return linalg.norm(self.X - self.Yprv)

    def eval_objfn(self):
        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]

    def obfn_dfd(self):
        return self.eval_Rx()

    def obfn_reg(self):
        reg = linalg.norm(self.X.ravel(), 1)
        return (self.lmbda*reg, reg)

    def getmin(self):
        """Reimplement getmin func to have unified output layout."""
        # [K, m, N] -> [N, K, m] -> [H, W, 1, K, m]
        minimizer = super().getmin()
        minimizer = minimizer.transpose((2, 0, 1))
        minimizer = minimizer.reshape(self.cri.shpX)
        return minimizer

    def reconstruct(self, X=None):
        """Reconstruct representation.  The reconstruction follows standard
        output layout."""
        if X is None:
            X = self.X
        else:
            # X has standard shape of (N0, N1, ..., 1, K, M) since we use
            # single channel coefficient array, first convert to
            # (num_atoms, batch_size, ...) and then to
            # (slice_dim, batch_size, num_slices_per_batch) by multiplying D.
            # [ H, W, 1, K, m ] -> [K, m, H, W, 1] -> [K, m, N]
            X = X.transpose((3, 4, 0, 1, 2))
            X = X.reshape(X.shape[0], X.shape[1], -1)
        recon = self.slices2im(np.matmul(self.D, X))
        # [K, C, H, W] -> [H, W, C, K, 1]
        recon = np.expand_dims(recon.transpose((2, 3, 1, 0)), axis=-1)
        return recon

    def setdict(self, D):
        """Set dictionary properly."""
        if D.ndim == 2:  # [patch_size, num_atoms]
            self.D = D.copy()
        elif D.ndim == 3:  # [patch_h, patch_w, num_atoms]
            self.D = D.reshape((-1, D.shape[-1]))
        elif D.ndim == 4:  # [patch_h, patch_w, channels, num_atoms]
            assert D.shape[-2] == 1 or D.shape[-2] == 3
            self.D = D.transpose(2, 0, 1, 3)
            self.D = self.D.reshape((-1, self.D.shape[-1]))
        else:
            raise ValueError('Invalid dict D dimension of {}'.format(D.shape))

    def getcoef(self):
        return self.X

    def im2slices(self, S):
        kernel_h, kernel_w = self.cri.shpD[:2]
        return im2slices(S, kernel_h, kernel_w, self.boundary)

    def slices2im(self, slices):
        kernel_h, kernel_w = self.cri.shpD[:2]
        output_h, output_w = self.cri.shpS[:2]
        return slices2im(slices, kernel_h, kernel_w, output_h, output_w,
                         self.boundary)


class ConvCnstrMODSliceFISTA(fista.FISTA):

    class Options(fista.FISTA.Options):
        defaults = copy.deepcopy(fista.FISTA.Options.defaults)
        defaults.update({
            'Boundary': 'circulant_back',
            'ZeroMean': True,
        })

        def __init__(self, opt=None):
            if opt is None:
                opt = {}
            super().__init__(opt)

    itstat_fields_objfn = ('DFid', 'Cnstr')
    hdrtxt_objfn = ('DFid', 'Cnstr')
    hdrval_objfun = {'DFid': 'DFid', 'Cnstr': 'Cnstr'}

    def __init__(self, Z, S, dsz, opt=None):
        """
        Parameters
        ----------
        Z: numpy array, [K, m, M]
            Learned sparse representations.
        S: numpy array, [K, C, H, W]
            Signal array.
        dsz: tuple
            The shape of the dictionary, in (kernel_h, kernel_w, channels,
            num_atoms).  Note that this tuple is only used to get access to
            patch size. The internal dictionary is NOT in this size.
        """
        if opt is None:
            opt = ConvCnstrMODSliceFISTA.Options()
        self.kernel_h, self.kernel_w = dsz[:2]
        xshape = (np.prod(dsz[:-1]), dsz[-1])
        self.set_attr('boundary', opt['Boundary'], dval='circulant_back',
                      dtype=None)
        super().__init__(np.prod(xshape), xshape, S.dtype, opt)
        self.S = np.asarray(S, dtype=self.dtype)
        if Z is not None:
            self.setcoef(Z)
        self.Y = self.X.copy()

    def eval_grad(self):
        recon = self.slices2im(np.matmul(self.Y, self.Z))
        slices = self.im2slices(recon - self.S)
        grad = np.tensordot(slices, self.Z, axes=((0, 2), (0, 2)))
        return grad

    def eval_proxop(self, V):
        return self.Pcn(V)

    def eval_R(self, V):
        recon = self.slices2im(np.matmul(V, self.Z))
        return linalg.norm(recon-self.S) ** 2 / 2.

    def rsdl(self):
        return linalg.norm(self.X - self.Yprv)

    def setcoef(self, Z):
        self.Z = np.asarray(Z, dtype=self.dtype)

    def getdict(self):
        return self.X

    def im2slices(self, S):
        return im2slices(S, self.kernel_h, self.kernel_w, self.boundary)

    def slices2im(self, slices):
        output_h, output_w = self.S.shape[-2:]
        return slices2im(slices, self.kernel_h, self.kernel_w, output_h,
                         output_w, self.boundary)

    def Pcn(self, D):
        return Pcn(D, self.opt['ZeroMean'])

    def eval_objfn(self):
        dfd = self.obfn_dfd()
        cns = self.obfn_cns()
        return (dfd, cns)

    def obfn_dfd(self):
        return self.eval_Rx()

    def obfn_cns(self):
        return linalg.norm(self.Pcn(self.X) - self.X)

    def reconstruct(self, D=None):
        if D is None:
            D = self.X
        recon = self.slices2im(np.matmul(D, self.Z))
        # [K, C, H, W] -> [H, W, C, K, 1]
        recon = np.expand_dims(recon.transpose((2, 3, 1, 0)), axis=-1)
        return recon


class ConvBPDNSliceDictLearnFISTA(dictlrn.DictLearn):

    class Options(dictlrn.DictLearn.Options):
        defaults = copy.deepcopy(dictlrn.DictLearn.Options.defaults)
        defaults.update({
            'AccurateDFid': False,
            'CBPDN': copy.deepcopy(ConvBPDNSliceFISTA.Options.defaults),
            'CCMOD': copy.deepcopy(ConvCnstrMODSliceFISTA.Options.defaults)
        })
        defaults['CBPDN'].update({'MaxMainIter': 1})
        defaults['CCMOD'].update({'MaxMainIter': 1})

        def __init__(self, opt=None):
            super().__init__({
                'CBPDN': ConvBPDNSliceFISTA.Options(self.defaults['CBPDN']),
                'CCMOD': ConvCnstrMODSliceFISTA.Options(self.defaults['CCMOD'])
            })
            if opt is None:
                opt = {}
            self.update(opt)

    def __init__(self, D0, S, lmbda=None, opt=None, dimK=None, dimN=2):
        if opt is None:
            opt = ConvBPDNSliceDictLearnFISTA.Options()
        self.opt = opt

        D0 = Pcn(D0, opt['CCMOD', 'ZeroMean'])
        xstep = ConvBPDNSliceFISTA(D0, S, lmbda, opt['CBPDN'],
                                   dimK=dimK, dimN=dimN)
        opt['CCMOD', 'X0'] = xstep.D
        dstep = ConvCnstrMODSliceFISTA(None, xstep.S, D0.shape, opt['CCMOD'])
        isc = self.config_itstats()
        super().__init__(xstep, dstep, opt, isc)

    def config_itstats(self):
        """Config itstats output for fista."""
        # isfld
        isfld = ['Iter', 'ObjFun', 'DFid', 'RegL1', 'Cnstr']
        if self.opt['CBPDN', 'BackTrack', 'Enabled']:
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
        if self.opt['CBPDN', 'BackTrack', 'Enabled']:
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
        if self.opt['CBPDN', 'BackTrack', 'Enabled']:
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
        return dictlrn.IterStatsConfig(
            isfld=isfld,
            isxmap=isxmap,
            isdmap=isdmap,
            evlmap=evlmap,
            hdrtxt=hdrtxt,
            hdrmap=hdrmap,
            fmtmap=fmtmap
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
        D = self.dstep.getdict().copy()
        patch_h, patch_w = self.xstep.cri.shpD[:2]
        D = D.reshape(-1, patch_h, patch_w, D.shape[-1])
        assert D.shape[0] == 1 or D.shape[0] == 3
        # transpose the channel dimension back
        D = D.transpose(1, 2, 0, 3)
        return D

    def reconstruct(self, D=None, X=None):
        """Reconstruct representation."""
        if D is None:
            D = self.getdict()
        elif D.ndim == 3:
            D = np.expand_dims(D, axis=-2)
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
            X = self.xstep.X
            recon = self.xstep.slices2im(np.matmul(D, X))
            dfd = ((recon - self.xstep.S) ** 2).sum() / 2.0
            reg = np.sum(np.abs(X))
            return dict(DFid=dfd, RegL1=reg, ObjFun=dfd+self.xstep.lmbda*reg)
        return None

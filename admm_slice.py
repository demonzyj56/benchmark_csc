"""Solve CSC using ADMM on slices."""
import copy
import logging
import pyfftw  # pylint: disable=unused-import
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
from sporco.admm import admm, bpdn, cbpdn
import sporco.cnvrep as cr
import sporco.linalg as sl
import sporco.util as su
from sporco.util import u

logger = logging.getLogger(__name__)


def _pad_circulant_front(blob, pad_h, pad_w):
    """Pad a 4-D blob with circulant boundary condition at the front."""
    return np.pad(blob, ((0, 0), (0, 0), (pad_h, 0), (pad_w, 0)), 'wrap')


def _pad_circulant_back(blob, pad_h, pad_w):
    """Pad a 4-D blob with circulant boundary condition at the back."""
    return np.pad(blob, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 'wrap')


def _pad_zeros_front(blob, pad_h, pad_w):
    """Pad a 4-D blob with zero boundary condition at the front."""
    return np.pad(blob, ((0, 0), (0, 0), (pad_h, 0), (pad_w, 0)), 'constant')


def _pad_zeros_back(blob, pad_h, pad_w):
    """Pad a 4-D blob with zero boundary condition at the back."""
    return np.pad(blob, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 'constant')


def _crop_circulant_front(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for circulant boundary condition
    at the front."""
    cropped = blob[:, :, pad_h:, pad_w:]
    cropped[:, :, -pad_h:, :] += blob[:, :, :pad_h, pad_w:]
    cropped[:, :, :, -pad_w:] += blob[:, :, pad_h:, :pad_w]
    cropped[:, :, -pad_h:, -pad_w:] += blob[:, :, :pad_h, :pad_w]
    return cropped


def _crop_circulant_back(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for circulant boundary condition
    at the back."""
    cropped = blob[:, :, :-pad_h, :-pad_w]
    cropped[:, :, :pad_h, :] += blob[:, :, -pad_h:, :-pad_w]
    cropped[:, :, :, :pad_w] += blob[:, :, :-pad_h, -pad_w:]
    cropped[:, :, :pad_h, :pad_w] += blob[:, :, -pad_h:, -pad_w:]
    return cropped


def _crop_zeros_front(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for zero boundary condition
    at the front."""
    return blob[:, :, pad_h:, pad_w:]


def _crop_zeros_back(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for zero boundary condition
    at the back."""
    return blob[:, :, :-pad_h, :-pad_w]


def _iter_recorder(solver):
    """Print the iteration information in logger when not in verbose mode."""
    if not solver.opt['Verbose']:
        if len(solver.itstat) == 0:
            elapsed_time = 0.
        elif len(solver.itstat) == 1:
            elapsed_time = solver.itstat[0].Time
        else:
            elapsed_time = solver.itstat[-1].Time - solver.itstat[-2].Time
        logger.info(
            'Iteration %d/%d, elapsed time: %.3fs',
            solver.k+1, solver.opt['MaxMainIter'], elapsed_time
        )
    return 0


class ConvBPDNSliceTwoBlockCnstrnt(admm.ADMMTwoBlockCnstrnt):

    class Options(admm.ADMMTwoBlockCnstrnt.Options):
        defaults = copy.deepcopy(admm.ADMMTwoBlockCnstrnt.Options.defaults)
        defaults.update({
            'RelaxParam': 1.8,
            'AuxVarObj': False,
            'Boundary': 'circulant_back',
            'RhoRatio': 1.,
            'Callback': _iter_recorder,
        })
        defaults['AutoRho'].update({
            'Enabled': True,
            'AutoScaling': True,
            'Period': 1,
            'Scaling': 1000.,  # tau
            'RsdlRatio': 1.2,  # mu
            'RsdlTarget': None,  # xi, initial value depends on lambda
        })

        def __init__(self, opt=None):
            if opt is None:
                opt = {}
            super().__init__(opt)

    # Although we split the variables in a different way, we record
    # the same objection function for comparison
    # follows exactly as cbpdn.ConvBPDN for actual comparison
    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), 'XStepAvg', 'YStepAvg')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1',
                     'XStepAvg': 'XStepTime', 'YStepAvg': 'YStepTime'}
    # timer for xstep/ystep
    # NOTE: You can't use Timer to measure each tic time.  Record average
    # time instead.
    itstat_fields_extra = ('XStepTime', 'YStepTime')

    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
        if opt is None:
            opt = ConvBPDNSliceTwoBlockCnstrnt.Options()
        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)
        if not hasattr(self, 'cri'):
            self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
        self.lmbda = self.dtype.type(lmbda)
        # Set penalty parameter if not set
        self.set_attr('rho', opt['rho'], dval=(50.0*self.lmbda + 1.0),
                      dtype=self.dtype)
        # Set xi if not set
        self.set_attr('tau_xi', opt['AutoRho', 'RsdlTarget'],
                      dval=(1.0+18.3**(np.log10(self.lmbda)+1.0)),
                      dtype=self.dtype)
        # set boundary condition
        self.set_attr('boundary', opt['Boundary'], dval='circulant_back',
                      dtype=None)
        # set rho ratio between two constraints
        self.set_attr('rho_ratio', opt['RhoRatio'], dval=0.5, dtype=self.dtype)
        self.setdict(D)
        # Number of elements of sparse representation x is invariant to
        # slice/FFT solvers.
        Nx = np.product(self.cri.shpX)
        # Externally the input signal should have a data layout as
        # S(N0, N1, ..., C, K).
        # First convert to common pytorch Variable layout.
        # [H, W, C, K, 1] -> [K, C, H, W]
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=S.dtype)
        self.S = self.S.squeeze(-1).transpose((3, 2, 0, 1))
        # [K, n, N]
        self.S_slice = self.im2slices(self.S)
        yshape = list(self.S_slice.shape)
        yshape[1] = self.D.shape[0] + self.D.shape[1]  # n+m
        super().__init__(Nx, yshape, 1, self.D.shape[0], S.dtype, opt)
        self.extra_timer = su.Timer(['xstep', 'ystep'])

    def setdict(self, D):
        """Set dictionary properly."""
        self.D = D.reshape(-1, D.shape[-1])
        self.lu, self.piv = sl.lu_factor(self.D, self.rho_ratio)
        self.lu = np.asarray(self.lu, dtype=self.dtype)

    def im2slices(self, S):
        r"""Convert the input signal :math:`S` to a slice form.
        Assuming the input signal having a standard shape as pytorch variable
        (N, C, H, W).  The output slices have shape
        (batch_size, slice_dim, num_slices_per_batch).
        """
        kernel_size = self.cri.shpD[:2]
        pad_h, pad_w = kernel_size[0] - 1, kernel_size[1] - 1
        S_torch = globals()['_pad_{}'.format(self.boundary)](
            S, pad_h, pad_w
        )
        with torch.no_grad():
            S_torch = torch.from_numpy(S_torch)
            slices = F.unfold(S_torch, kernel_size=kernel_size)
        return slices.numpy()

    def slices2im(self, slices):
        r"""Reconstruct input signal :math:`\hat{S}` for slices.
        The input slices should have compatible size of
        (batch_size, slice_dim, num_slices_per_batch), and the
        returned signal has shape (N, C, H, W) as standard pytorch variable.
        """
        kernel_size = self.cri.shpD[:2]
        pad_h, pad_w = kernel_size[0] - 1, kernel_size[1] - 1
        output_h, output_w = self.cri.shpS[:2]
        with torch.no_grad():
            slices_torch = torch.from_numpy(slices)
            S_recon = F.fold(
                slices_torch, (output_h+pad_h, output_w+pad_w), kernel_size
            )
        S_recon = globals()['_crop_{}'.format(self.boundary)](
            S_recon.numpy(), pad_h, pad_w
        )
        return S_recon

    def yinit(self, yshape):
        """Initialization for variable Y."""
        Y = super().yinit(yshape)
        self._Y0, self._Y1 = self.block_sep(Y)
        return Y

    def uinit(self, ushape):
        """Initialization for variable U."""
        U = super().uinit(ushape)
        self._U0, self._U1 = self.block_sep(U)
        return U

    def xstep(self):
        self.extra_timer.start('xstep')
        rhs = self.cnst_A0T(self._Y0-self._U0) + \
            self.rho_ratio*self.cnst_A1T(self._Y1-self._U1)
        if self.X is None:
            self.X = np.zeros_like(rhs, dtype=self.dtype)
        # NOTE:
        # Here we solve the sparse code X for each batch index i, since X is
        # organized as [batch_size, num_atoms, num_signals].  This interface
        # may be modified for online setting.
        for i in range(self.X.shape[0]):
            self.X[i] = np.asarray(
                sl.lu_solve_ATAI(self.D, self.rho_ratio, rhs[i], self.lu, self.piv),
                dtype=self.dtype
            )
        self.extra_timer.stop('xstep')

    def relax_AX(self):
        self._AX0nr, self._AX1nr = self.cnst_A0(self.X), self.cnst_A1(self.X)
        if self.rlx == 1.0:
            self._AX0, self._AX1 = self._AX0nr, self._AX1nr
        else:
            # c0 and c1 are all zero
            alpha = self.rlx
            self._AX0 = alpha*self._AX0nr + (1-alpha)*self._Y0
            self._AX1 = alpha*self._AX1nr + (1-alpha)*self._Y1
        self.AXnr = self.block_cat(self._AX0nr, self._AX1nr)
        self.AX = self.block_cat(self._AX0, self._AX1)

    def ystep(self):
        self.extra_timer.start('ystep')
        p = self.S_slice / self.rho + self._AX0 + self._U0
        recon = self.slices2im(p)
        self._Y0[:] = p - self.im2slices(recon) / (p.shape[1] + self.rho)
        self._Y1[:] = sl.shrink1(self._AX1+self._U1, self.lmbda/self.rho/self.rho_ratio)
        self.Y = self.block_cat(self._Y0, self._Y1)
        self.extra_timer.stop('ystep')

    def ustep(self):
        self._U0 += self._AX0 - self._Y0
        self._U1 += self._AX1 - self._Y1
        self.U = self.block_cat(self._U0, self._U1)

    def rsdl_r(self, AX, Y):
        return AX - Y

    def cnst_A0(self, X):
        return np.matmul(self.D, X)

    def cnst_A1(self, X):
        return X

    def cnst_A0T(self, Y0):
        return np.matmul(self.D.transpose(), Y0)

    def cnst_A1T(self, Y1):
        return Y1

    def var_y0(self):
        return self._Y0

    def var_y1(self):
        return self._Y1

    def getmin(self):
        """Reimplement getmin func to have unified output layout."""
        # [K, m, N] -> [N, K, m] -> [H, W, 1, K, m]
        minimizer = super().getmin()
        minimizer = minimizer.transpose((2, 0, 1))
        minimizer = minimizer.reshape(self.cri.shpX)
        return minimizer

    def eval_objfn(self):
        """Overwrite this function as in ConvBPDN."""
        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]

    def obfn_dfd(self):
        r"""Data fidelity term of the objective :math:`(1/2) \|s - \sum_i
        \mathbf{R}_i^T y_i\|_2^2`.
        """
        recon = self.slices2im(self.cnst_A0(self.X))
        return ((recon - self.S) ** 2).sum() / 2.0

    def obfn_reg(self):
        r"""Regularization term of the objective :math:`g(y)=\|y\|_1`.
        Returns a tuple where the first is the scaled combination of all
        regularization terms (if exist) and the sesequent ones are each term.
        """
        l1 = linalg.norm(self.X.ravel(), 1)
        return (self.lmbda * l1, l1)

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

    def update_rho(self, k, r, s):
        """Back to usual way of updating rho."""
        if self.opt['AutoRho', 'AutoScaling']:
            # If AutoScaling is enabled, use adaptive penalty parameters by
            # residual balancing as commonly used in SPORCO.
            super().update_rho(k, r, s)
        else:
            tau = self.rho_tau
            mu = self.rho_mu
            if k != 0 and ((k+1) % self.opt['AutoRho', 'Period'] == 0):
                if r > mu * s:
                    self.rho = tau * self.rho
                elif s > mu * r:
                    self.rho = self.rho / tau

    def itstat_extra(self):
        """Get xstep/ystep solve time. Return average time."""
        niters = self.k + 1
        xstep_elapsed = self.extra_timer.elapsed('xstep')
        ystep_elapsed = self.extra_timer.elapsed('ystep')
        return (xstep_elapsed/niters, ystep_elapsed/niters)


class ConvBPDNSlice(admm.ADMM):
    r"""Slice-based convolutional sparse coding solver using ADMM.
    This method is detailed in [1]. In specific, it solves the CSC problem
    in the following form:

    .. math::
        \min_{x_i,y_i} \frac{1}{2}\|s-\sum_i\mathbf{R}_i^T y_i\|_2^2
        + \lambda\sum_i \|x_i\|_1 \;\mathrm{suth\;that}\;
        y_i = D_l x_i\;\forall i.

    If we let :math:`g(y)=\frac{1}{2}\|s-\sum_i\mathbf{R}_i^Ty_i\|_2^2`,
    :math:`f(x)=\lambda\sum_i\|x_i\|_1`, then the objective can be
    updated using ADMM.

    [1] V. Papyan, Y. Romano, J. Sulam, and M. Elad, “Convolutional Dictionary
        Learning via Local Processing,” arXiv:1705.03239 [cs], May 2017.
    """

    class Options(admm.ADMM.Options):
        """Slice-based convolutional sparse coding options.
        Options include all fields of :class:`admm.cbpdn.ConvBPDN`,
        with `BPDN` from :class:`admm.bpdn.BPDN`.
        """
        defaults = copy.deepcopy(admm.ADMM.Options.defaults)
        defaults.update({
            'BPDN': copy.deepcopy(bpdn.BPDN.Options.defaults),
            'RelaxParam': 1.8,
            'Boundary': 'circulant_back',
        })
        defaults['BPDN'].update({
            'MaxMainIter': 1000,
            'Verbose': False,
        })
        defaults['AutoRho'].update({
            'Enabled': True,
            'AutoScaling': True,
            'Period': 1,
            'Scaling': 1000.,  # tau
            'RsdlRatio': 1.2,  # mu
            'RsdlTarget': None,  # xi, initial value depends on lambda
        })

        def __init__(self, opt=None):
            super().__init__({'BPDN': bpdn.BPDN.Options()})
            if opt is None:
                opt = {}
            self.update(opt)

    # follows exactly as cbpdn.ConvBPDN for actual comparison
    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'), 'XStepAvg', 'YStepAvg')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1',
                     'XStepAvg': 'XStepTime', 'YStepAvg': 'YStepTime'}
    # timer for xstep/ystep
    # NOTE: You can't use Timer to measure each tic time.  Record average
    # time instead.
    itstat_fields_extra = ('XStepTime', 'YStepTime')

    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
        r"""We use the same layout as ConvBPDN as input and output, but for
        internal computation we use a differnt layout.

        Internal Parameters
        -------------------
        X: [K, m, N]
          Convolutional representation of the input signal. m is the size
          of atom in a dictionary, K is the batch size of input signals,
          and N is the number of slices extracted from each signal (usually
          number of pixels in an image).
        Y: [K, n, N]
          Splitted variable with contraint :math:`D_l x_i - y_i = 0`.
          n represents the size of each slice.
        U: [K, n, N]
          Dual variable with the same size as Y.
        """
        if opt is None:
            opt = ConvBPDNSlice.Options()
        # Set dtype attribute based on S.dtype and opt['DataType']
        self.set_dtype(opt, S.dtype)
        if not hasattr(self, 'cri'):
            self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
        self.boundary = opt['Boundary']
        # Number of elements of sparse representation x is invariant to
        # slice/FFT solvers.
        Nx = np.product(self.cri.shpX)
        # (N, M)
        self.D = D.reshape((-1, D.shape[-1]))
        # Externally the input signal should have a data layout as
        # S(N0, N1, ..., C, K).
        # First convert to common pytorch Variable layout.
        # [H, W, C, K, 1] -> [K, C, H, W]
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=S.dtype)
        self.S = self.S.squeeze(-1).transpose((3, 2, 0, 1))
        # [K, n, N]
        self.S_slice = self.im2slices(self.S)
        self.lmbda = lmbda
        # Set penalty parameter if not set
        self.set_attr('rho', opt['rho'], dval=(50.0*self.lmbda + 1.0),
                      dtype=self.dtype)
        # Set xi if not set
        self.set_attr('tau_xi', opt['AutoRho', 'RsdlTarget'],
                      dval=(1.0+18.3**(np.log10(self.lmbda)+1.0)),
                      dtype=self.dtype)
        super().__init__(Nx, self.S_slice.shape, self.S_slice.shape,
                         S.dtype, opt)
        self.extra_timer = su.Timer(['xstep', 'ystep'])

    def im2slices(self, S):
        r"""Convert the input signal :math:`S` to a slice form.
        Assuming the input signal having a standard shape as pytorch variable
        (N, C, H, W).  The output slices have shape
        (batch_size, slice_dim, num_slices_per_batch).
        """
        # NOTE: we simulate the boundary condition outside fold and unfold.
        kernel_size = self.cri.shpD[:2]
        pad_h, pad_w = kernel_size[0] - 1, kernel_size[1] - 1
        S_torch = globals()['_pad_{}'.format(self.boundary)](S, pad_h, pad_w)
        with torch.no_grad():
            S_torch = torch.from_numpy(S_torch)
            slices = F.unfold(S_torch, kernel_size=kernel_size)
        assert slices.size(1) == self.D.shape[0]
        return slices.numpy()

    def slices2im(self, slices):
        r"""Reconstruct input signal :math:`\hat{S}` for slices.
        The input slices should have compatible size of
        (batch_size, slice_dim, num_slices_per_batch), and the
        returned signal has shape (N, C, H, W) as standard pytorch variable.
        """
        kernel_size = self.cri.shpD[:2]
        pad_h, pad_w = kernel_size[0] - 1, kernel_size[1] - 1
        output_h, output_w = self.cri.shpS[:2]
        with torch.no_grad():
            slices_torch = torch.from_numpy(slices)
            S_recon = F.fold(
                slices_torch, (output_h+pad_h, output_w+pad_w), kernel_size
            )
        S_recon = globals()['_crop_{}'.format(self.boundary)](
            S_recon.numpy(), pad_h, pad_w
        )
        return S_recon

    def xstep(self):
        r"""Minimize with respect to :math:`x`.  This has the form:

        .. math::
            f(x)=\sum_i\left(\lambda\|x_i\|_1+\frac{\rho}{2}\|D_l x_i - y_i
            + u_i\|_2^2\right).

        This could be solved in parallel over all slice indices i and all
        batch indices k (implicit in the above form).
        """
        self.extra_timer.start('xstep')
        # TODO(leoyolo): The naive method here is to apply BPDN once and
        # destroy the object.
        signal = self.Y - self.U
        signal = signal.transpose((1, 0, 2))
        # [K, n, N] -> [n, K, N] -> [n, K*N]
        signal = signal.reshape(signal.shape[0], -1)
        opt = copy.deepcopy(self.opt['BPDN'])
        opt['Y0'] = getattr(self, '_X_bpdn_cache', None)
        solver = bpdn.BPDN(self.D, signal, lmbda=self.lmbda/self.rho, opt=opt)
        self.X = solver.solve()
        self._X_bpdn_cache = copy.deepcopy(self.X)
        self.X = self.X.reshape(
            self.X.shape[0], self.Y.shape[0], self.Y.shape[2]
        ).transpose((1, 0, 2))
        self.extra_timer.stop('xstep')

    def ystep(self):
        r"""Minimize with respect to :math:`y`.  This has the form:

        .. math::
            g(y)=\frac{1}{2}\|s-\sum_i\mathbf{R}_i^T y_i\|_2^2+
            \frac{\rho}{2}\sum_i\|D_l x_i - y_i + u_i\|_2^2.

        This has a very nice solution

        .. math::
            p_i=\frac{1}{\rho}\mathbf{R}_i s + D_l x_i + u_i.
        .. math::
            \hat{s}=\sum_i \mathbf{R}_i^T p_i.
        .. math::
            y_i = p_i - \frac{1}{\rho+n}\mathbf{R}_i\hat{s}.

        """
        self.extra_timer.start('ystep')
        # Notice that AX = D*X.
        p = self.S_slice / self.rho + self.AX + self.U
        recon = self.slices2im(p)
        self.Y = p - self.im2slices(recon) / (p.shape[1] + self.rho)
        self.extra_timer.stop('ystep')

    def cnst_A(self, X):
        r"""Compute :math:`Ax`. Our constraint is

        ..math::
            D_l x_i - y_i = 0
        """
        return np.matmul(self.D, X)

    def cnst_AT(self, X):
        r"""Compute :math:`A^T x`. Our constraint is

        ..math::
            D_l x_i - y_i = 0
        """
        return np.matmul(self.D.transpose(), X)

    def cnst_B(self, Y):
        r""" Compute :math:`By`.  Our constraint is

        ..math::
            D_l x_i - y_i = 0
        """
        return -Y

    def cnst_c(self):
        r""" Compute :math:`c`.  Our constraint is

        ..math::
            D_l x_i - y_i = 0
        """
        return 0.

    def yinit(self, yshape):
        """Slices are initialized using signal slices."""
        _ = yshape
        y_init = self.S_slice.copy()
        y_init /= y_init.shape[1]
        return y_init

    def getmin(self):
        """Reimplement getmin func to have unified output layout."""
        # [K, m, N] -> [N, K, m] -> [H, W, 1, K, m]
        minimizer = self.X.copy()
        minimizer = minimizer.transpose((2, 0, 1))
        minimizer = minimizer.reshape(self.cri.shpX)
        return minimizer

    def eval_objfn(self):
        """Overwrite this function as in ConvBPDN."""
        dfd = self.obfn_dfd()
        reg = self.obfn_reg()
        obj = dfd + reg[0]
        return (obj, dfd) + reg[1:]

    def obfn_dfd(self):
        r"""Data fidelity term of the objective :math:`(1/2) \|s - \sum_i
        \mathbf{R}_i^T y_i\|_2^2`.
        """
        # notice AX = D*X
        # use non-relaxed version to represent data fidelity term
        recon = self.slices2im(self.cnst_A(self.X))
        return ((recon - self.S) ** 2).sum() / 2.0

    def obfn_reg(self):
        r"""Regularization term of the objective :math:`g(y)=\|y\|_1`.
        Returns a tuple where the first is the scaled combination of all
        regularization terms (if exist) and the sesequent ones are each term.
        """
        l1 = linalg.norm(self.X.ravel(), 1)
        return (self.lmbda * l1, l1)

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

    def update_rho(self, k, r, s):
        """Back to usual way of updating rho."""
        if self.opt['AutoRho', 'AutoScaling']:
            # If AutoScaling is enabled, use adaptive penalty parameters by
            # residual balancing as commonly used in SPORCO.
            super().update_rho(k, r, s)
        else:
            tau = self.rho_tau
            mu = self.rho_mu
            if k != 0 and ((k+1) % self.opt['AutoRho', 'Period'] == 0):
                if r > mu * s:
                    self.rho = tau * self.rho
                elif s > mu * r:
                    self.rho = self.rho / tau

    def itstat_extra(self):
        """Get xstep/ystep solve time. Return average time."""
        niters = self.k + 1
        xstep_elapsed = self.extra_timer.elapsed('xstep')
        ystep_elapsed = self.extra_timer.elapsed('ystep')
        return (xstep_elapsed/niters, ystep_elapsed/niters)

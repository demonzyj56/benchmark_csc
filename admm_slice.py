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
from sporco.util import u

logger = logging.getLogger(__name__)


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
        defaults = copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)
        defaults.update({
            'BPDN': copy.deepcopy(bpdn.BPDN.Options.defaults)
        })
        defaults['BPDN'].update({
            'MaxMainIter': 1,
            'Verbose': False,
        })

        def __init__(self, opt=None):
            if opt is None:
                opt = {}
            super().__init__(opt)

    # follows exactly as cbpdn.ConvBPDN for actual comparison
    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', u('Regℓ1'): 'RegL1'}

    def __init__(self, D, S, lmbda=None, opt=None, dimK=None, dimN=2):
        r"""We use the same layout as ConvBPDN as input and output, but for
        internal computation we use a differnt layout.

        Internal Parameters
        -------------------
        X: [m, K, N]
          Convolutional representation of the input signal. m is the size
          of atom in a dictionary, K is the batch size of input signals,
          and N is the number of slices extracted from each signal (usually
          number of pixels in an image).
        Y: [n, K, N]
          Splitted variable with contraint :math:`y_i=D_l x_i`. n represents
          the size of each slice.
        U: [n, K, N]
          Dual variable with the same size as Y.
        """
        if opt is None:
            opt = ConvBPDNSlice.Options()
        if not hasattr(self, 'cri'):
            self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)
        # Number of elements of sparse representation x is invariant to
        # slice/FFT solvers.
        Nx = np.product(self.cri.shpX)
        # (N, M)
        self.D = D.reshape((-1, D.shape[-1]))
        # Externally the input signal should have a data layout as
        # S(N0, N1, ..., C, K).
        # First convert to common pytorch Variable layout.
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=S.dtype)
        self.S = self.S.squeeze(-1).transpose((3, 2, 0, 1))
        # (N, K)
        self.S_slice = self.im2slices(self.S)
        super().__init__(Nx, self.S_slice.shape, self.S_slice.shape,
                         S.dtype, opt)
        self.lmbda = lmbda

    def im2slices(self, S):
        r"""Convert the input signal :math:`S` to a slice form.
        Assuming the input signal having a standard shape as pytorch variable
        (N, C, H, W).  The output slices have shape
        (slice_dim, batch_size, num_slices_per_batch).
        """
        # NOTE: we use zero-padding as boundary condition for now.
        # Handling boundary condition remains a critical work.
        # TODO(leoyolo): Gurantee that F.unfold and F.fold with zero padding
        # behaves as expected, by writing some tests.
        # TODO(leoyolo): Handle different boundary condition.
        S_torch = torch.from_numpy(S)
        kernel_size = self.cri.shpD[:2]
        pad_h, pad_w = kernel_size[0]//2, kernel_size[1]//2
        with torch.no_grad():
            slices = F.unfold(S_torch, kernel_size=kernel_size,
                              padding=(pad_h, pad_w))
        assert slices.size(1) == self.D.shape[0]
        slices = slices.permute(1, 0, 2).numpy()
        return slices

    def slices2im(self, slices):
        r"""Reconstruct input signal :math:`\hat{S}` for slices.
        The input slices should have compatible size of
        (slice_dim, batch_size, num_slices_per_batch), and the
        returned signal has shape (N, C, H, W) as standard pytorch variable.
        """
        output_size = self.cri.shpS[:2]
        kernel_size = self.cri.shpD[:2]
        pad_h, pad_w = kernel_size[0]//2, kernel_size[1]//2
        slices_torch = torch.from_numpy(slices)
        slices_torch = slices_torch.permute(1, 0, 2)
        with torch.no_grad():
            S_recon = F.fold(
                slices_torch, output_size, kernel_size, padding=(pad_h, pad_w)
            ).numpy()
        return S_recon

    def xstep(self):
        r"""Minimize with respect to :math:`x`.  This has the form:

        .. math::
            f(x)=\sum_i\left(\lambda\|x_i\|_1+\frac{\rho}{2}\|y_i
            - D_l x_i + u_i\|_2^2\right).

        This could be solved in parallel over all slice indices i and all
        batch indices k (implicit in the above form).
        """
        # TODO(leoyolo): The naive method here is to apply BPDN once and
        # destroy the object.
        signal = self.Y + self.U
        signal = signal.reshape(signal.shape[0], -1)
        solver = bpdn.BPDN(self.D, signal, lmbda=self.lmbda/self.rho,
                           opt=self.opt['BPDN'])
        self.X = solver.solve()
        self.X = self.X.reshape(self.X.shape[0], self.Y.shape[1],
                                self.Y.shape[2])

    def ystep(self):
        r"""Minimize with respect to :math:`y`.  This has the form:

        .. math::
            g(y)=\frac{1}{2}\|s-\sum_i\mathbf{R}_i^T y_i\|_2^2+
            \frac{\rho}{2}\sum_i\|y_i-D_l x_i + u_i\|_2^2.

        This has a very nice solution

        .. math::
            p_i=\frac{1}{\rho}\mathbf{R}_i s + D_l x_i - u_i.
        .. math::
            \hat{s}=\sum_i \mathbf{R}_i^T p_i.
        .. math::
            y_i = p_i - \frac{1}{\rho+n}\mathbf{R}_i\hat{s}.

        """
        p = self.S_slice / self.rho + self.AX - self.U
        recon = self.slices2im(p)
        self.Y = p - self.im2slices(recon) / (p.shape[0] + self.rho)

    def cnst_A(self, X):
        r"""Compute :math:`Ax`. Our constraint is

        ..math::
            y_i - D_l x_i = 0
        """
        return -np.matmul(self.D, X)

    def cnst_AT(self, X):
        r"""Compute :math:`A^T x`. Our constraint is

        ..math::
            y_i - D_l x_i = 0
        """
        return -np.matmul(self.D.transpose(), X)

    def cnst_B(self, Y):
        r""" Compute :math:`By`.  Our constraint is

        ..math::
            y_i - D_l x_i = 0
        """
        return Y

    def cnst_c(self):
        r""" Compute :math:`c`.  Our constraint is

        ..math::
            y_i - D_l x_i = 0
        """
        return 0.

    def yinit(self, yshape):
        """Slices are initialized using signal slices."""
        _ = yshape
        y_init = self.S_slice.copy()
        y_init /= y_init.shape[0]
        return y_init

    def getmin(self):
        """Reimplement getmin func to have unified output layout."""
        # (m, K, N)
        minimizer = self.X.copy()
        minimizer = minimizer.transpose((2, 1, 0))
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
        recon = self.slices2im(np.matmul(self.D, self.X))
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
            X = self.X.transpose((4, 3, 0, 1, 2))
            X = X.reshape(X.shape[0], X.shape[1], -1)
        recon = self.slices2im(np.matmul(self.D, X))
        recon = np.expand_dims(recon.transpose((2, 3, 1, 0)), axis=-1)
        return recon

## This file will be used to contain all code we need from Kymatio
## Can tidy up the organisation once we've extracted all the code we need
import torch
import torch.nn as nn
from torch.nn import ReflectionPad2d ## Eventually want to get rid of this

from sn_camels.scattering.torch_backend import backend

## Utils imports
import scipy.fftpack
import warnings


###########################################################################################################
######## Filter Bank ########
###########################################################################################################

import numpy as np
from scipy.fftpack import fft2, ifft2


def filter_bank(M, N, J, L=8):
    """
        Builds in Fourier the Morlet filters used for the scattering transform.
        Each single filter is provided as a dictionary with the following keys:
        * 'j' : scale
        * 'theta' : angle used
        Parameters
        ----------
        M, N : int
            spatial support of the input
        J : int
            logscale of the scattering
        L : int, optional
            number of angles used for the wavelet transform
        Returns
        -------
        filters : list
            A two list of dictionary containing respectively the low-pass and
             wavelet filters.
        Notes
        -----
        The design of the filters is optimized for the value L = 8.
    """
    filters = {}
    filters['psi'] = []

    for j in range(J):
        for theta in range(L):
            psi = {}
            psi['j'] = j
            psi['theta'] = theta
            psi_signal = morlet_2d(M, N, 0.8 * 2**j,
                (int(L-L/2-1)-theta) * np.pi / L,
                3.0 / 4.0 * np.pi /2**j, 4.0/L)
            psi_signal_fourier = fft2(psi_signal)
            # drop the imaginary part, it is zero anyway
            psi_signal_fourier = np.real(psi_signal_fourier)
            for res in range(min(j + 1, max(J - 1, 1))):
                psi_signal_fourier_res = periodize_filter_fft(
                    psi_signal_fourier, res)
                psi[res] = psi_signal_fourier_res
            filters['psi'].append(psi)

    filters['phi'] = {}
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0)
    phi_signal_fourier = fft2(phi_signal)
    # drop the imaginary part, it is zero anyway
    phi_signal_fourier = np.real(phi_signal_fourier)
    filters['phi']['j'] = J
    for res in range(J):
        phi_signal_fourier_res = periodize_filter_fft(phi_signal_fourier, res)
        filters['phi'][res] = phi_signal_fourier_res

    return filters


def periodize_filter_fft(x, res):
    """
        Parameters
        ----------
        x : numpy array
            signal to periodize in Fourier
        res :
            resolution to which the signal is cropped.
        Returns
        -------
        crop : numpy array
            It returns a crop version of the filter, assuming that
             the convolutions will be done via compactly supported signals.
    """
    M = x.shape[0]
    N = x.shape[1]

    crop = np.zeros((M // 2 ** res, N // 2 ** res), x.dtype)

    mask = np.ones(x.shape, np.float32)
    len_x = int(M * (1 - 2 ** (-res)))
    start_x = int(M * 2 ** (-res - 1))
    len_y = int(N * (1 - 2 ** (-res)))
    start_y = int(N * 2 ** (-res - 1))
    mask[start_x:start_x + len_x,:] = 0
    mask[:, start_y:start_y + len_y] = 0
    x = np.multiply(x,mask)

    for k in range(int(M / 2 ** res)):
        for l in range(int(N / 2 ** res)):
            for i in range(int(2 ** res)):
                for j in range(int(2 ** res)):
                    crop[k, l] += x[k + i * int(M / 2 ** res), l + j * int(N / 2 ** res)]

    return crop


def morlet_2d(M, N, sigma, theta, xi, slant=0.5, offset=0):
    """
        Computes a 2D Morlet filter.
        A Morlet filter is the sum of a Gabor filter and a low-pass filter
        to ensure that the sum has exactly zero mean in the temporal domain.
        It is defined by the following formula in space:
        psi(u) = g_{sigma}(u) (e^(i xi^T u) - beta)
        where g_{sigma} is a Gaussian envelope, xi is a frequency and beta is
        the cancelling parameter.
        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts
        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    wv = gabor_2d(M, N, sigma, theta, xi, slant, offset)
    wv_modulus = gabor_2d(M, N, sigma, theta, 0, slant, offset)
    K = np.sum(wv) / np.sum(wv_modulus)

    mor = wv - K * wv_modulus
    return mor


def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.
        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts
        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)

    norm_factor = (2 * 3.1415 * sigma * sigma / slant)
    gab /= norm_factor

    return gab

###########################################################################################################
######## Utils ########
###########################################################################################################
def compute_padding(M, N, J):
    """
         Precomputes the future padded size. If 2^J=M or 2^J=N,
         border effects are unavoidable in this case, and it is
         likely that the input has either a compact support,
         either is periodic.
         Parameters
         ----------
         M, N : int
             input size
         Returns
         -------
         M, N : int
             padded size
    """
    M_padded = ((M + 2 ** J) // 2 ** J + 1) * 2 ** J
    N_padded = ((N + 2 ** J) // 2 ** J + 1) * 2 ** J

    return M_padded, N_padded

def fft2(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return scipy.fftpack.fft2(x)



##############################################################################################################
######################## Scattering classes  ########################
##############################################################################################################

class ScatteringTorch(nn.Module):
    def __init__(self):
        super(ScatteringTorch, self).__init__()
        self.frontend_name = 'torch'

    def register_filters(self):
        """ This function should be called after filters are generated,
        saving those arrays as module buffers. """
        raise NotImplementedError

    def scattering(self, x):
        """ This function should compute the scattering transform."""
        raise NotImplementedError

    def forward(self, x):
        """This method is an alias for `scattering`."""

        input_checks(x)

        return self.scattering(x)


class ScatteringBase():
    def __init__(self):
        super(ScatteringBase, self).__init__()

    def build(self):
        """ Defines elementary routines.
        This function should always call and create the filters via
        self.create_filters() defined below. For instance, via:
        self.filters = self.create_filters() """
        raise NotImplementedError

    def _instantiate_backend(self, import_string):
        """ This function should instantiate the backend to be used if not already
        specified"""

        # Either the user entered a string, in which case we load the corresponding backend.
        if isinstance(self.backend, str):
            if self.backend.startswith(self.frontend_name):
                try:
                    print(import_string, self.backend , "_backend")
                    #self.backend = importlib.import_module(import_string + self.backend + "_backend", 'backend').backend
                    self.backend = backend
                except ImportError:
                    raise ImportError('Backend ' + self.backend + ' not found!')
            else:
                raise ImportError('The backend ' + self.backend + ' can not be called from the frontend ' +
                                   self.frontend_name + '.')
        # Either the user passed a backend object, in which case we perform a compatibility check.
        else:
            if not self.backend.name.startswith(self.frontend_name):
                raise ImportError('The backend ' + self.backend.name + ' is not supported by the frontend ' +
                                   self.frontend_name + '.')

    def create_filters(self):
        """ This function should run a filterbank function that
        will create the filters as numpy array, and then, it should
        save those arrays. """
        raise NotImplementedError


class ScatteringBase2D(ScatteringBase):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend=None, out_type='array'):
        super(ScatteringBase2D, self).__init__()
        self.pre_pad = pre_pad
        self.L = L
        self.backend = backend
        self.J = J
        self.shape = shape
        self.max_order = max_order
        self.out_type = out_type

    def build(self):
        self.M, self.N = self.shape

        if 2 ** self.J > self.M or 2 ** self.J > self.N:
            raise RuntimeError('The smallest dimension should be larger than 2^J.')
        self.M_padded, self.N_padded = compute_padding(self.M, self.N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        if not self.pre_pad:
            self.pad = self.backend.Pad([(self.M_padded - self.M) // 2, (self.M_padded - self.M+1) // 2, (self.N_padded - self.N) // 2,
                                (self.N_padded - self.N + 1) // 2], [self.M, self.N])
        else:
            self.pad = lambda x: x

        self.unpad = self.backend.unpad

    def create_filters(self):
        filters = filter_bank(self.M, self.N, self.J, self.L)
        self.phi, self.psi = filters['phi'], filters['psi']

class ScatteringTorch2D(ScatteringTorch, ScatteringBase2D):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend='torch', out_type='array'):
        ScatteringTorch.__init__(self)
        ScatteringBase2D.__init__(**locals())
        ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2D.build(self)
        ScatteringBase2D.create_filters(self)

        if pre_pad:
            # Need to cast to complex in Torch
            self.pad = lambda x: x.reshape(x.shape + (1,))

        self.register_filters()

    def register_single_filter(self, v, n):
        current_filter = torch.from_numpy(v).unsqueeze(-1)
        self.register_buffer('tensor' + str(n), current_filter)
        return current_filter

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters

        n = 0

        for c, phi in self.phi.items():
            if not isinstance(c, int):
                continue

            self.phi[c] = self.register_single_filter(phi, n)
            n = n + 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if not isinstance(k, int):
                    continue

                self.psi[j][k] = self.register_single_filter(v, n)
                n = n + 1

    def load_single_filter(self, n, buffer_dict):
        return buffer_dict['tensor' + str(n)]

    def load_filters(self):
        """ This function loads filters from the module's buffers """
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        buffer_dict = dict(self.named_buffers())

        n = 0

        phis = self.phi
        for c, phi in phis.items():
            if not isinstance(c, int):
                continue

            phis[c] = self.load_single_filter(n, buffer_dict)
            n = n + 1

        psis = self.psi
        for j in range(len(psis)):
            for k, v in psis[j].items():
                if not isinstance(k, int):
                    continue

                psis[j][k] = self.load_single_filter(n, buffer_dict)
                n = n + 1

        return phis, psis

    def scattering(self, input):
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')

        if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))

        if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        phi, psi = self.load_filters()

        batch_shape = input.shape[:-2]
        signal_shape = input.shape[-2:]

        input = input.reshape((-1,) + signal_shape)

        S = scattering2d(input, self.pad, self.unpad, self.backend, self.J,
                            self.L, phi, psi, self.max_order, self.out_type)

        if self.out_type == 'array':
            scattering_shape = S.shape[-3:]
            S = S.reshape(batch_shape + scattering_shape)
        else:
            scattering_shape = S[0]['coef'].shape[-2:]
            new_shape = batch_shape + scattering_shape

            for x in S:
                x['coef'] = x['coef'].reshape(new_shape)

        return S



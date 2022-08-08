## This file will be used to contain all code we need from Kymatio
## Can tidy up the organisation once we've extracted all the code we need
import torch
import torch.nn as nn
from torch.nn import ReflectionPad2d ## Eventually want to get rid of this

from sn_camels.scattering import torch_backend, create_filters

## Utils imports
import scipy.fftpack
import warnings


###########################################################################################################
######## Filter Bank ########
###########################################################################################################

import numpy as np
from scipy.fftpack import fft2, ifft2




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
                    #self.backend = importlib.import_module(import_string + self.backend + "_backend", 'backend').backend
                    self.backend = torch_backend.backend
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
        #self.phi=get_psis(self.M, self.N, self.J, self.L)
        #self.psi=get_phis(self.M, self.N, self.J, self.L)

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



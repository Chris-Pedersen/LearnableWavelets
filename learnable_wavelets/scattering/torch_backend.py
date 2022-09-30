import torch
from collections import namedtuple
from torch.autograd import Function
from packaging import version

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

class FFT:
    def __init__(self, fft, ifft, irfft, type_checks):
        self.fft = fft
        self.ifft = ifft
        self.irfft = irfft
        self.sanity_checks = type_checks

    def fft_forward(self, x, direction='C2C', inverse=False):
        """Interface with FFT routines for any dimensional signals and any backend signals.
            Example (for Torch)
            -------
            x = torch.randn(128, 32, 32, 2)
            x_fft = fft(x)
            x_ifft = fft(x, inverse=True)
            Parameters
            ----------
            x : input
                Complex input for the FFT.
            direction : string
                'C2R' for complex to real, 'C2C' for complex to complex.
            inverse : bool
                True for computing the inverse FFT.
                NB : If direction is equal to 'C2R', then an error is raised.
            Raises
            ------
            RuntimeError
                In the event that we are going from complex to real and not doing
                the inverse FFT or in the event x is not contiguous.
            Returns
            -------
            output :
                Result of FFT or IFFT.
        """
        if direction == 'C2R':
            if not inverse:
                raise RuntimeError('C2R mode can only be done with an inverse FFT.')

        self.sanity_checks(x)

        if direction == 'C2R':
            output = self.irfft(x)
        elif direction == 'C2C':
            if inverse:
                output = self.ifft(x)
            else:
                output = self.fft(x)

        return output

    def __call__(self, x, direction='C2C', inverse=False):
        return self.fft_forward(x, direction=direction, inverse=inverse)

def input_checks(x):
    if x is None:
        raise TypeError('The input should be not empty.')

    if not x.is_contiguous():
        raise RuntimeError('The input must be contiguous.')

def _is_complex(x):
    return x.shape[-1] == 2

def _is_real(x):
    return x.shape[-1] == 1

class ModulusStable(Function):
    """Stable complex modulus
    This class implements a modulus transform for complex numbers which is
    stable with respect to very small inputs (z close to 0), avoiding
    returning nans in all cases.
    Usage
    -----
    modulus = ModulusStable.apply  # apply inherited from Function
    x_mod = modulus(x)
    Parameters
    ---------
    x : tensor
        The complex tensor (i.e., whose last dimension is two) whose modulus
        we want to compute.
    Returns
    -------
    output : tensor
        A tensor of same size as the input tensor, except for the last
        dimension, which is removed. This tensor is differentiable with respect
        to the input in a stable fashion (so gradent of the modulus at zero is
        zero).
    """

    @staticmethod
    def forward(ctx, x):
        """Forward pass of the modulus.
        This is a static method which does not require an instantiation of the
        class.
        Arguments
        ---------
        ctx : context object
            Collected during the forward pass. These are automatically added
            by PyTorch and should not be touched. They are then used for the
            backward pass.
        x : tensor
            The complex tensor whose modulus is to be computed.
        Returns
        -------
        output : tensor
            This contains the modulus computed along the last axis, with that
            axis removed.
        """
        ctx.p = 2
        ctx.dim = -1
        ctx.keepdim = False

        output = (x[...,0] * x[...,0] + x[...,1] * x[...,1]).sqrt()

        ctx.save_for_backward(x, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the modulus
        This is a static method which does not require an instantiation of the
        class.
        Arguments
        ---------
        ctx : context object
            Collected during the forward pass. These are automatically added
            by PyTorch and should not be touched. They are then used for the
            backward pass.
        grad_output : tensor
            The gradient with respect to the output tensor computed at the
            forward pass.
        Returns
        -------
        grad_input : tensor
            The gradient with respect to the input.
        """
        x, output = ctx.saved_tensors
        if ctx.dim is not None and ctx.keepdim is False and x.dim() != 1:
            grad_output = grad_output.unsqueeze(ctx.dim)
            output = output.unsqueeze(ctx.dim)

        grad_input = x.mul(grad_output).div(output)

        # Special case at 0 where we return a subgradient containing 0
        grad_input.masked_fill_(output == 0, 0)

        return grad_input

# shortcut for ModulusStable.apply
modulus = ModulusStable.apply

class Modulus():
    """This class implements a modulus transform for complex numbers.
        Usage
        -----
        modulus = Modulus()
        x_mod = modulus(x)
        Parameters
        ---------
        x : tensor
            Complex torch tensor.
        Returns
        -------
        output : tensor
            A tensor with the same dimensions as x, such that output[..., 0]
            contains the complex modulus of x, while output[..., 1] = 0.
    """
    def __call__(self, x):
        type_checks(x)

        norm = torch.zeros_like(x)
        norm[..., 0] = modulus(x)
        return norm

def type_checks(x):
    if not _is_complex(x):
        raise TypeError('The input should be complex (i.e. last dimension is 2).')

    if not x.is_contiguous():
        raise RuntimeError('Tensors must be contiguous.')

def cdgmm(A, B, inplace=False):
    """Complex pointwise multiplication.
        Complex pointwise multiplication between (batched) tensor A and tensor B.
        Parameters
        ----------
        A : tensor
            A is a complex tensor of size (B, C, M, N, 2).
        B : tensor
            B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1).
        inplace : boolean, optional
            If set to True, all the operations are performed in place.
        Raises
        ------
        RuntimeError
            In the event that the filter B is not a 3-tensor with a last
            dimension of size 1 or 2, or A and B are not compatible for
            multiplication.
        TypeError
            In the event that A is not complex, or B does not have a final
            dimension of 1 or 2, or A and B are not of the same dtype, or if
            A and B are not on the same device.
        Returns
        -------
        C : tensor
            Output tensor of size (B, C, M, N, 2) such that:
            C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :].
    """
    if not _is_real(B):
        type_checks(B)
    else:
        if not B.is_contiguous():
            raise RuntimeError('Tensors must be contiguous.')

    type_checks(A)

    if A.shape[-len(B.shape):-1] != B.shape[:-1]:
        raise RuntimeError('The filters are not compatible for multiplication.')

    if A.dtype is not B.dtype:
        raise TypeError('Input and filter must be of the same dtype.')

    if B.device.type == 'cuda':
        if A.device.type == 'cuda':
            if A.device.index != B.device.index:
                raise TypeError('Input and filter must be on the same GPU.')
        else:
            raise TypeError('Input must be on GPU.')

    if B.device.type == 'cpu':
        if A.device.type == 'cuda':
            raise TypeError('Input must be on CPU.')

    if _is_real(B):
        if inplace:
            return A.mul_(B)
        else:
            return A * B
    else:
        C = A.new(A.shape)

        A_r = A[..., 0].view(-1, B.nelement() // 2)
        A_i = A[..., 1].view(-1, B.nelement() // 2)

        B_r = B[..., 0].view(-1).unsqueeze(0).expand_as(A_r)
        B_i = B[..., 1].view(-1).unsqueeze(0).expand_as(A_i)

        C[..., 0].view(-1, B.nelement() // 2)[:] = A_r * B_r - A_i * B_i
        C[..., 1].view(-1, B.nelement() // 2)[:] = A_r * B_i + A_i * B_r

        return C if not inplace else A.copy_(C)

def concatenate(arrays, dim):
    return torch.stack(arrays, dim=dim)


def real(x):
    """Real part of complex tensor
    Takes the real part of a complex tensor, where the last axis corresponds
    to the real and imaginary parts.
    Parameters
    ----------
    x : tensor
        A complex tensor (that is, whose last dimension is equal to 2).
    Returns
    -------
    x_real : tensor
        The tensor x[..., 0] which is interpreted as the real part of x.
    """
    return x[..., 0]



class Pad(object):
    def __init__(self, pad_size, input_size):
        """Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.
            Parameters
            ----------
            pad_size : list of 4 integers
                Size of padding to apply [top, bottom, left, right].
            input_size : list of 2 integers
                size of the original signal [height, width].
        """
        self.pad_size = pad_size
        self.input_size = input_size

        self.build()

    def build(self):
        """Builds the padding module.
            Attributes
            ----------
            padding_module : ReflectionPad2d
                Pads the input tensor using the reflection of the input
                boundary.
        """
        pad_size_tmp = list(self.pad_size)

        # This handles the case where the padding is equal to the image size
        if pad_size_tmp[0] == self.input_size[0]:
            pad_size_tmp[0] -= 1
            pad_size_tmp[1] -= 1
        if pad_size_tmp[2] == self.input_size[1]:
            pad_size_tmp[2] -= 1
            pad_size_tmp[3] -= 1
        # Pytorch expects its padding as [left, right, top, bottom]
        self.padding_module = ReflectionPad2d([pad_size_tmp[2], pad_size_tmp[3],
                                               pad_size_tmp[0], pad_size_tmp[1]])

    def __call__(self, x):
        """Applies padding and maps to complex.
            Parameters
            ----------
            x : tensor
                Real tensor input to be padded and sent to complex domain.
            Returns
            -------
            output : tensor
                Complex torch tensor that has been padded.
        """
        batch_shape = x.shape[:-2]
        signal_shape = x.shape[-2:]
        x = x.reshape((-1, 1) + signal_shape)
        x = self.padding_module(x)

        # Note: PyTorch is not effective to pad signals of size N-1 with N
        # elements, thus we had to add this fix.
        if self.pad_size[0] == self.input_size[0]:
            x = torch.cat([x[:, :, 1, :].unsqueeze(2), x, x[:, :, x.shape[2] - 2, :].unsqueeze(2)], 2)
        if self.pad_size[2] == self.input_size[1]:
            x = torch.cat([x[:, :, :, 1].unsqueeze(3), x, x[:, :, :, x.shape[3] - 2].unsqueeze(3)], 3)

        output = x.new_zeros(x.shape + (2,))
        output[..., 0] = x
        output = output.reshape(batch_shape + output.shape[-3:])
        return output


def unpad(in_):
    """Unpads input.
        Slices the input tensor at indices between 1:-1.
        Parameters
        ----------
        in_ : tensor
            Input tensor.
        Returns
        -------
        in_[..., 1:-1, 1:-1] : tensor
            Output tensor.  Unpadded input.
    """
    return in_[..., 1:-1, 1:-1]

class SubsampleFourier(object):
    """Subsampling of a 2D image performed in the Fourier domain
        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.
        Parameters
        ----------
        x : tensor
            Input tensor with at least 5 dimensions, the last being the real
            and imaginary parts.
        k : int
            Integer such that x is subsampled by k along the spatial variables.
        Returns
        -------
        out : tensor
            Tensor such that its Fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k].
    """
    def __call__(self, x, k):
        if not _is_complex(x):
            raise TypeError('The x should be complex.')

        if not x.is_contiguous():
            raise RuntimeError('Input should be contiguous.')
        batch_shape = x.shape[:-3]
        signal_shape = x.shape[-3:]
        x = x.view((-1,) + signal_shape)
        y = x.view(-1,
                       k, x.shape[1] // k,
                       k, x.shape[2] // k,
                       2)

        out = y.mean(3, keepdim=False).mean(1, keepdim=False)
        out = out.reshape(batch_shape + out.shape[-3:])
        return out

if version.parse(torch.__version__) >= version.parse('1.8'):
    fft = FFT(lambda x: torch.view_as_real(torch.fft.fft2(torch.view_as_complex(x))),
          lambda x: torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(x))),
          lambda x: torch.fft.ifft2(torch.view_as_complex(x)).real,
          type_checks)
else:
    fft = FFT(lambda x: torch.fft(x, 2, normalized=False),
              lambda x: torch.ifft(x, 2, normalized=False),
              lambda x: torch.irfft(x, 2, normalized=False, onesided=False),
              type_checks)

backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'concatenate'])
backend.name = 'torch'
backend.version = torch.__version__
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = fft
backend.Pad = Pad
backend.unpad = unpad
backend.concatenate = lambda x: concatenate(x, -3)
# Authors: Chris Pedersen
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna, Edouard Oyallon, Muawiz Chaudhary
## Originally taken from https://github.com/kymatio/kymatio/blob/master/kymatio/scattering2d/core/scattering2d.py on 1 March 2022

import torch

def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array'):
    """ Function to take an input image and perform a series of scattering
    convolutions."""
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []

    ## Map to complex
    complex_maps = x.new_zeros(x.shape + (2,))
    complex_maps[..., 0] = x

    U_0_c = fft(complex_maps, 'C2C')

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi[0])
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    S_0 = fft(U_1_c, 'C2R', inverse=True)

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'theta': ()})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1][0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = fft(U_1_c, 'C2C', inverse=True)
        U_1_c = modulus(U_1_c)
        U_1_c = fft(U_1_c, 'C2C')

        # Second low pass filter
        S_1_c = cdgmm(U_1_c, phi[j1])
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

        S_1_r = fft(S_1_c, 'C2R', inverse=True)

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'theta': (theta1,)})

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1:
                continue

            U_2_c = cdgmm(U_1_c, psi[n2][j1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = fft(U_2_c, 'C2C', inverse=True)
            U_2_c = modulus(U_2_c)
            U_2_c = fft(U_2_c, 'C2C')

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi[j2])
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            S_2_r = fft(S_2_c, 'C2R', inverse=True)

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        out_S = concatenate([x['coef'] for x in out_S])

    return out_S

def scattering2d_learn(x, pad, unpad, backend, J, L, phi, psi, max_order,
        split_filters, subsample ,out_type='array'):
    """ Function to take an input image and perform a series of scattering
    convolutions."""
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []
    
    ## Map to complex
    complex_maps = x.new_zeros(x.shape + (2,))
    complex_maps[..., 0] = x

    U_0_c = fft(complex_maps, 'C2C')

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi[0])
    U_1_c = subsample_fourier(U_1_c, k=subsample)


    S_0 = fft(U_1_c, 'C2R', inverse=True)

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'theta': ()})

    if split_filters:
        for n1 in range(int(len(psi)/2)):
            j1 = psi[n1]['j'] ## don't care about this any more
            theta1 = psi[n1]['theta']

            ## Wavelet convolution
            U_1_c = cdgmm(U_0_c, psi[n1][0])

            U_1_c = fft(U_1_c, 'C2C', inverse=True)
            U_1_c = modulus(U_1_c)
            U_1_c = fft(U_1_c, 'C2C')

            ## Second low pass filter
            S_1_c = cdgmm(U_1_c, phi[0])
            S_1_c = subsample_fourier(S_1_c, k=subsample)

            S_1_r = fft(S_1_c, 'C2R', inverse=True)

            out_S_1.append({'coef': S_1_r,
                            'j': (j1,),
                            'theta': (theta1,)})

            if max_order < 2:
                continue
            for n2 in range(int(len(psi)/2),len(psi)):
                j2 = psi[n2]['j']
                theta2 = psi[n2]['theta']
                

                U_2_c = cdgmm(U_1_c, psi[n2][0])
                U_2_c = fft(U_2_c, 'C2C', inverse=True)
                U_2_c = modulus(U_2_c)
                U_2_c = fft(U_2_c, 'C2C')

                ## Low pass filter
                S_2_c = cdgmm(U_2_c, phi[0])
                
                S_2_c = subsample_fourier(S_2_c, k=subsample)

                S_2_r = fft(S_2_c, 'C2R', inverse=True)
                

                out_S_2.append({'coef': S_2_r,
                                'j': (j1, j2),
                                'theta': (theta1, theta2)})
    else:
        for n1 in range(len(psi)):
            j1 = psi[n1]['j'] ## don't care about this any more
            theta1 = psi[n1]['theta']

            ## Wavelet convolution
            U_1_c = cdgmm(U_0_c, psi[n1][0])

            U_1_c = fft(U_1_c, 'C2C', inverse=True)
            U_1_c = modulus(U_1_c)
            U_1_c = fft(U_1_c, 'C2C')

            ## Second low pass filter
            S_1_c = cdgmm(U_1_c, phi[0])
            S_1_c = subsample_fourier(S_1_c, k=subsample)

            S_1_r = fft(S_1_c, 'C2R', inverse=True)

            out_S_1.append({'coef': S_1_r,
                            'j': (j1,),
                            'theta': (theta1,)})

            if max_order < 2:
                continue
            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                theta2 = psi[n2]['theta']
                

                U_2_c = cdgmm(U_1_c, psi[n2][0])
                U_2_c = fft(U_2_c, 'C2C', inverse=True)
                U_2_c = modulus(U_2_c)
                U_2_c = fft(U_2_c, 'C2C')

                ## Low pass filter
                S_2_c = cdgmm(U_2_c, phi[0])
                S_2_c = subsample_fourier(S_2_c, k=subsample)
                S_2_r = fft(S_2_c, 'C2R', inverse=True)
                

                out_S_2.append({'coef': S_2_r,
                                'j': (j1, j2),
                                'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        out_S = concatenate([x['coef'] for x in out_S])

    return out_S

def construct_scattering(input, scattering, psi, learnable, split_filters, subsample):
    """ Construct the scattering object

        Parameters:
            input      -- input data
            scattering -- Kymatio (https://www.kymat.io/) scattering object
            psi        -- dictionnary of filters that is used in the kymatio code
            learnable  -- are we using learnable filters
            split_filters -- split first and second order filters
        Returns:
            S -- output of the scattering network
    """
    if not torch.is_tensor(input):
        raise TypeError('The input should be a PyTorch Tensor.')

    if len(input.shape) < 2:
        raise RuntimeError('Input tensor must have at least two dimensions.')

    if not input.is_contiguous():
        raise RuntimeError('Tensor must be contiguous.')

    if (input.shape[-1] != scattering.N or input.shape[-2] != scattering.M) and not scattering.pre_pad:
        raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (scattering.M, scattering.N))

    if not scattering.out_type in ('array', 'list'):
        raise RuntimeError("The out_type must be one of 'array' or 'list'.")

    batch_shape = input.shape[:-2]
    signal_shape = input.shape[-2:]

    input = input.reshape((-1,) + signal_shape)

    if learnable:
        S = scattering2d_learn(input, scattering.pad, scattering.unpad, scattering.backend, scattering.J,
                        scattering.L, scattering.phi, psi, scattering.max_order, split_filters, subsample,
                        scattering.out_type)
    else:
        S = scattering2d(input, scattering.pad, scattering.unpad, scattering.backend, scattering.J,
                        scattering.L, scattering.phi, psi, scattering.max_order, scattering.out_type)

    if scattering.out_type == 'array':
        scattering_shape = S.shape[-3:]
        S = S.reshape(batch_shape + scattering_shape)

    return S

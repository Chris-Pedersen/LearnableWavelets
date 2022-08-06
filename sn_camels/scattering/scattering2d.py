# Authors: Chris Pedersen
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna, Edouard Oyallon, Muawiz Chaudhary
## Originally taken from https://github.com/kymatio/kymatio/blob/master/kymatio/scattering2d/core/scattering2d.py on 1 March 2022

import torch

def do_convolutions(x, backend, J, L, phi, psi, max_order,
        split_filters, subsample):
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

    out_S = concatenate([x['coef'] for x in out_S])

    return out_S

def convolve_fields(input, backend, J, L, phi, psi, max_order, split_filters, subsample):
    """  
        Wrapper function for a loop that will convovle each wavelet with the input fields

        Parameters:
            input      -- input data
            psi        -- dictionnary of filters that is used in the kymatio code
            learnable  -- are we using learnable filters
            split_filters -- split first and second order filters
        Returns:
            S -- Fields after being convolved with wavelets
    """

    batch_shape = input.shape[:-2]
    signal_shape = input.shape[-2:]

    input = input.reshape((-1,) + signal_shape)

    S = do_convolutions(input, backend, J, L, phi, psi,
                        max_order, split_filters, subsample)

    ## S will always be a numpy array
    scattering_shape = S.shape[-3:]
    S = S.reshape(batch_shape + scattering_shape)

    return S



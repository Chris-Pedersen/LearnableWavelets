"""Helper functions for regenerating the scattering filters on the fly

Authors: Benjamin Therien, Shanel Gauthier, Laurent Alsene-Racicot, Michael Eickenberg

Functions: 
    construct_scattering         -- Construct the scattering object
    update_psi                   -- Update the psi dictionnary with the new wavelets
    get_total_num_filters        -- Compute the total number of filters
    periodize_filter_fft         -- Periodize the filter in fourier space
    create_filters_params_random -- Create reusable randomly initialized filter parameters:
                                    orientations, xis, sigmas, sigmas
    create_filters_params        -- Create reusable tight frame initialized filter parameters: 
                                    orientations, xis, sigmas, sigmas
    raw_morlets                  -- Helper function for creating morlet filters 
    morlets                      -- Creates morlet wavelet filters from inputs

"""

import sys
from pathlib import Path 
import numpy as np
from sn_camels.scattering import kymat_code
sys.path.append(str(Path.cwd()))
import torch
import scipy.fftpack
import warnings

def initialise_filters(M,N,num_first_order,num_second_order,initialization,seed,device,
                                                                requires_grad,use_cuda):
    """ Initialise wavelet filters """

    ## Set up grid to generate filters
    shape = (scattering.M, scattering.N,)
    ranges = [torch.arange(-(s // 2), -(s // 2) + s, device=device, dtype=torch.float) for s in shape]
    grid = torch.stack(torch.meshgrid(*ranges), 0).to(device)
    params_filters_first = create_filters_params_random(num_first_order,requires_grad,device,seed)
    params_filters_second = create_filters_params_random(num_second_order,requires_grad,device,seed)

    wavelets_first  = morlets(shape, params_filters_first[0], params_filters_first[1], 
                    params_filters_first[2], params_filters_first[3], device=device)
    wavelets_second  = morlets(shape, params_filters_second[0], params_filters_second[1], 
                    params_filters_second[2], params_filters_second[3], device=device)




def create_scatteringExclusive(J,N,M,max_order,device,initialization,seed=0,requires_grad=True,use_cuda=True):
    """Creates scattering parameters and replaces then with the specified initialization

    Creates the scattering network, adds it to the passed device, and returns it for modification. Next,
    based on input, we modify the filter initialization and use it to create new conv kernels. Then, we
    update the psi Kymatio object to match the Kymatio API.

    arguments:
    use_cuda -- True if were using gpu
    J -- scale of scattering (always 2 for now)
    N -- height of the input image
    M -- width of the input image
    initilization -- the type of init: ['Tight-Frame' or 'Random']
    seed -- the seed used for creating randomly initialized filters
    requires_grad -- boolean idicating whether we want to learn params
    """

    params_filters = []

    if initialization == "Tight-Frame":
        params_filters = create_filters_params(J,L,requires_grad,device) #kymatio init
    elif initialization == "Random":
        num_filters= J*8 ## temporarily hardcoded
        params_filters = create_filters_params_random(num_filters,requires_grad,device,seed) #random init
    else:
        raise InvalidInitializationException

    shape = (M, N,)
    ranges = [torch.arange(-(s // 2), -(s // 2) + s, device=device, dtype=torch.float) for s in shape]
    grid = torch.stack(torch.meshgrid(*ranges), 0).to(device)
    params_filters =  [ param.to(device) for param in params_filters]

    wavelets  = morlets(shape, params_filters[0], params_filters[1], 
                    params_filters[2], params_filters[3], device=device)
    
    #psi = update_psi(J, psi, wavelets, device) #update psi to reflect the new conv filters

    #return scattering, psi, wavelets, params_filters, grid
    return wavelets, params_filters, grid

def update_psi(J, psi, wavelets, device):
    """ Update the psi dictionary with the new wavelets

        Parameters:
            J -- scale for the scattering
            psi -- dictionary of filters
            wavelets -- wavelet filters
            device -- device cuda or cpu

        Returns:
            psi -- dictionnary of filters
    """
    wavelets = wavelets.real.contiguous().unsqueeze(3)
    
    if J == 2:
        for i,d in enumerate(psi):
                d[0] = wavelets[i]

    else:
        for i,d in enumerate(psi):
            for res in range(0, J-1):
                if res in d.keys():
                    if res == 0:
                        d[res] = wavelets[i]
                    else:
                        d[res] = periodize_filter_fft(wavelets[i].squeeze(2), res).unsqueeze(2)
                
    return psi

def get_total_num_filters(J, L):
    """ Compute the total number of filters

        Parameters:
            J -- scale of the scattering
            L -- number of orientation for the scattering

        Returns:
            num_filters -- the total number of filters
    """
    num_filters = 0
    for j in range(2,J+1):
        num_filters += j *L
    return num_filters  

def periodize_filter_fft(x, res):
    """ Periodize the filter in fourier space

        Parameters:
            x -- signal to periodize in Fourier 
            res -- resolution to which the signal is cropped
            

        Returns:
            periodized -- It returns a crop version of the filter, assuming that the convolutions
                          will be done via compactly supported signals.           
    """

    s1, s2 = x.shape
    periodized = x.reshape(res*2, s1// 2**res, res*2, s2//2**res).mean(dim=(0,2))
    return periodized 

def periodize_filter_fft_kymat(x, res):
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

def create_filters_params_random(n_filters, is_scattering_dif, device, seed):
    """ Create reusable randomly initialized filter parameters: orientations, xis, sigmas, sigmas     

        Parameters:
            n_filters -- the number of filters in the filter bank
            is_scattering_dif -- boolean for the differentiability of the scattering
            device -- device cuda or cpu

        Returns:
            params -- list that contains the parameters of the filters
    """

    rng = np.random.RandomState(seed)
    orientations = rng.uniform(0,2*np.pi,n_filters) 
    slants = rng.uniform(0.5, 1.5,n_filters )
    xis = rng.uniform(0.5, 1, n_filters )
    sigmas = np.log(rng.uniform(np.exp(1), np.exp(5), n_filters ))
    
    xis = torch.tensor(xis, dtype=torch.float32, device=device)
    sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
    slants = torch.tensor(slants, dtype=torch.float32, device=device)
    orientations = torch.tensor(orientations, dtype=torch.float32, device=device)  
    params = [orientations, xis, sigmas, slants]

    if is_scattering_dif:
        for param in params:
            param.requires_grad = True

    return  params

def create_filters_params(J, L, is_scattering_dif, device):
    """ Create reusable tight frame initialized filter parameters: orientations, xis, sigmas, sigmas     

        Parameters:
            J -- scale of the scattering
            L -- number of orientation for the scattering
            is_scattering_dif -- boolean for the differentiability of the scattering
            device -- device cuda or cpu

        Returns:
            params -- list that contains the parameters of the filters

    """
    orientations = []
    xis = []
    sigmas = []
    slants = []

    for j in range(J):
        for theta in range(L):
            sigmas.append(0.8 * 2**j)
            t = ((int(L-L/2-1)-theta) * np.pi / L)
            xis.append(3.0 / 4.0 * np.pi /2**j)
            slant = 4.0/L
            slants.append(slant)
            orientations.append(t) 
     
    xis = torch.tensor(xis, dtype=torch.float32, device=device)
    sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
    slants = torch.tensor(slants, dtype=torch.float32, device=device)
    orientations = torch.tensor(orientations, dtype=torch.float32, device=device)  

    params = [orientations, xis, sigmas, slants]
    if is_scattering_dif:
        for param in params:
            param.requires_grad = True
    return  params


##########################################################################################################
########################################   PSN code  #####################################################
##########################################################################################################
def raw_morlets(grid_or_shape, wave_vectors, gaussian_bases, morlet=True, ifftshift=True, fft=True):
    """ Helper function for creating morlet filters

        Parameters:
            grid_or_shape -- a grid of the size of the filter or a tuple that indicates its shape
            wave_vectors -- directions of the wave part of the morlet wavelet
            gaussian_bases -- bases of the gaussian part of the morlet wavelet
            morlet -- boolean for morlet or gabor wavelet
            ifftshift -- boolean for the ifftshift (inverse fast fourier transform shift)
            fft -- boolean for the fft (fast fourier transform)

        Returns:
            filters -- the wavelet filters before normalization
            
    """ 
    n_filters, n_dim = wave_vectors.shape
    assert gaussian_bases.shape == (n_filters, n_dim, n_dim)
    device = wave_vectors.device

    if isinstance(grid_or_shape, tuple):
        shape = grid_or_shape
        ranges = [torch.arange(-(s // 2), -(s // 2) + s, device=device, dtype=torch.float) for s in shape]
        grid = torch.stack(torch.meshgrid(*ranges), 0)
    else:
        shape = grid_or_shape.shape[1:]
        grid = grid_or_shape

    waves = torch.exp(1.0j * torch.matmul(grid.T, wave_vectors.T).T)
    gaussian_directions = torch.matmul(grid.T, gaussian_bases.T.reshape(n_dim, n_dim * n_filters)).T
    gaussian_directions = gaussian_directions.reshape((n_dim, n_filters) + shape)
    radii = torch.norm(gaussian_directions, dim=0)
    gaussians = torch.exp(-0.5 * radii ** 2)
    signal_dims = list(range(1, n_dim + 1))
    gabors = gaussians * waves

    if morlet:
        gaussian_sums = gaussians.sum(dim=signal_dims, keepdim=True)
        gabor_sums = gabors.sum(dim=signal_dims, keepdim=True).real
        morlets = gabors - gabor_sums / gaussian_sums * gaussians
        filters = morlets
    else:
        filters = gabors
    if ifftshift:
        filters = torch.fft.ifftshift(filters, dim=signal_dims)
    if fft:
        filters = torch.fft.fftn(filters, dim=signal_dims)

    return filters

def morlets(grid_or_shape, theta, xis, sigmas, slants, device=None, morlet=True, ifftshift=True, fft=True):
    """Creates morlet wavelet filters from inputs

        Parameters:
            grid_or_shape -- a grid of the size of the filter or a tuple that indicates its shape
            theta -- global orientations of the wavelets
            xis -- frequency scales of the wavelets
            sigmas -- gaussian window scales of the wavelets
            slants -- slants of the wavelets
            device -- device cuda or cpu
            morlet -- boolean for morlet or gabor wavelet
            ifftshift -- boolean for the ifftshift (inverse fast fourier transform shift)
            fft -- boolean for the fft (fast fourier transform)

        Returns:
            wavelets -- the wavelet filters

    """
    if device is None:
        device = theta.device
    orientations = torch.cat((torch.cos(theta).unsqueeze(1),torch.sin(theta).unsqueeze(1)), dim =1)
    n_filters, ndim = orientations.shape
    wave_vectors = orientations * xis[:, np.newaxis]
    _, _, gauss_directions = torch.linalg.svd(orientations[:, np.newaxis])
    gauss_directions = gauss_directions / sigmas[:, np.newaxis, np.newaxis]
    indicator = torch.arange(ndim,device=device) < 1
    slant_modifications = (1.0 * indicator + slants[:, np.newaxis] * ~indicator)
    gauss_directions = gauss_directions * slant_modifications[:, :, np.newaxis]

    wavelets = raw_morlets(grid_or_shape, wave_vectors, gauss_directions, morlet=morlet, 
                          ifftshift=ifftshift, fft=fft)

    norm_factors = (2 * 3.1415 * sigmas * sigmas / slants).unsqueeze(1)

    if type(grid_or_shape) == tuple:
        norm_factors = norm_factors.expand([n_filters,grid_or_shape[0]]).unsqueeze(2).repeat(1,1,grid_or_shape[1])
    else:
        norm_factors = norm_factors.expand([n_filters,grid_or_shape.shape[1]]).unsqueeze(2).repeat(1,1,grid_or_shape.shape[2])

    wavelets = wavelets / norm_factors

    return wavelets


##########################################################################################################
###################################### kymatio code  #####################################################
##########################################################################################################

def fft2(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return scipy.fftpack.fft2(x)

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
                psi_signal_fourier_res = periodize_filter_fft_kymat(
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
        phi_signal_fourier_res = periodize_filter_fft_kymat(phi_signal_fourier, res)
        filters['phi'][res] = phi_signal_fourier_res

    return filters


def get_psis(M, N, J, L=8):
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
    psis = []

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
                psi_signal_fourier_res = periodize_filter_fft_kymat(
                    psi_signal_fourier, res)
                psi[res] = psi_signal_fourier_res
            psis.append(psi)
    return psis

def get_phis(M, N, J):
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
        Returns
        -------
        filters : list
            A two list of dictionary containing respectively the low-pass and
             wavelet filters.
        Notes
        -----
        The design of the filters is optimized for the value L = 8.
    """

    phis = []
    phi_signal = gabor_2d(M, N, 0.8 * 2**(J-1), 0, 0)
    phi_signal_fourier = fft2(phi_signal)
    # drop the imaginary part, it is zero anyway
    phi_signal_fourier = np.real(phi_signal_fourier)
    for res in range(J):
        phi_signal_fourier_res = periodize_filter_fft_kymat(phi_signal_fourier, res)
        print(phi_signal_fourier_res.shape)
        phis.append(phi_signal_fourier_res)

    return phis

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


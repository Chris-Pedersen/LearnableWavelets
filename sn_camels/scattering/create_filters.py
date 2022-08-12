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
from kymatio import Scattering2D
sys.path.append(str(Path.cwd()))
import torch

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
    scattering = Scattering2D(J=J, shape=(M, N), max_order=max_order, 
                              frontend='torch',pre_pad=True)

    L = scattering.L

    if use_cuda:
        scattering = scattering.cuda()

    phi, psi  = scattering.load_filters()
    
    params_filters = []

    if initialization == "Tight-Frame":
        params_filters = create_filters_params(J,L,requires_grad,device) #kymatio init
    elif initialization == "Random":
        num_filters= J*L
        params_filters = create_filters_params_random(num_filters,requires_grad,device,seed) #random init
    else:
        raise InvalidInitializationException

    shape = (scattering.M, scattering.N,)
    ranges = [torch.arange(-(s // 2), -(s // 2) + s, device=device, dtype=torch.float) for s in shape]
    grid = torch.stack(torch.meshgrid(*ranges), 0).to(device)
    params_filters =  [ param.to(device) for param in params_filters]

    wavelets, waves  = morlets(shape, params_filters[0], params_filters[1], 
                    params_filters[2], params_filters[3], device=device )
    
    psi = update_psi(J, psi, wavelets, device) #update psi to reflect the new conv filters

    return scattering, psi, wavelets, params_filters, grid, waves

def update_psi(J, psi, wavelets, device):
    """ Update the psi dictionnary with the new wavelets

        Parameters:
            J -- scale for the scattering
            psi -- dictionnary of filters
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
                        d[res] = periodize_filter_fft(wavelets[i].squeeze(2), res, device).unsqueeze(2)
                
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

def periodize_filter_fft(x, res, device):
    """ Periodize the filter in fourier space

        Parameters:
            x -- signal to periodize in Fourier 
            res -- resolution to which the signal is cropped
            device -- device cuda or cpu
            

        Returns:
            periodized -- It returns a crop version of the filter, assuming that the convolutions
                          will be done via compactly supported signals.           
    """

    s1, s2 = x.shape
    periodized = x.reshape(res*2, s1// 2**res, res*2, s2//2**res).mean(dim=(0,2))
    return periodized 

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

    return filters, waves

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

    wavelets, waves = raw_morlets(grid_or_shape, wave_vectors, gauss_directions, morlet=morlet, 
                          ifftshift=ifftshift, fft=fft)

    norm_factors = (2 * 3.1415 * sigmas * sigmas / slants).unsqueeze(1)

    if type(grid_or_shape) == tuple:
        norm_factors = norm_factors.expand([n_filters,grid_or_shape[0]]).unsqueeze(2).repeat(1,1,grid_or_shape[1])
    else:
        norm_factors = norm_factors.expand([n_filters,grid_or_shape.shape[1]]).unsqueeze(2).repeat(1,1,grid_or_shape.shape[2])

    wavelets = wavelets / norm_factors

    return wavelets, waves




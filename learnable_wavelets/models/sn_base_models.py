import torch
import torch.nn as nn

from learnable_wavelets.scattering import scattering2d, create_filters
from learnable_wavelets.models import models_utils
from learnable_wavelets.scattering import torch_backend

"""
"Base" models, which form the first (two) layers of the network and consist of wavelet convolutions
"""

class InvalidInitializationException(Exception):
    """Error thrown when an invalid initialization scheme is passed"""
    pass


class sn_Identity(nn.Module):
    """Identity nn.Module for identity"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n_coefficients = 1

    def forward(self, x):
        return x
        
    def saveFilterGrads(self,scatteringActive):
        pass

    def saveFilterValues(self,scatteringActive):
        pass

    def plotFilterGrad(self):
        pass

    def plotFilterGrads(self):
        pass

    def plotFilterValue(self):
        pass

    def plotFilterValues(self):
        pass

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        return 0
    
    def checkFilterDistance(self):
        return 0
    
    def setEpoch(self, epoch):
        self.epoch = epoch

    def releaseVideoWriters(self):
        pass
        
    def checkParamDistance(self):
        pass

    def checkDistance(self,compared):
        pass
    
    
class sn_ScatteringBase(nn.Module):
    """A learnable scattering nn.module """

    def __init__(self, J, N, M, channels, max_order, initialization, seed, 
                 device, learnable=True, lr_orientation=0.1, lr_scattering=0.1,
                 skip=True, split_filters=False, subsample=1, use_cuda=True,plot=True):
        """ Constructor for a scattering layer
        
        Creates scattering filters and adds them to the nn.parameters if learnable
        
        parameters: 
            J              -- Scale of scattering (always 2 for now - this parameter is being phased out)
            N              -- Height of the input image
            M              -- Width of the input image
            channels       -- Number of different input fields
            max_order      -- Highest order of wavelet scattering
            initilization  -- Wavelet initialisation ['Tight-Frame' or 'Random']
            seed           -- The random seed used to initialize the parameters
            device         -- The device to place weights on
            learnable      -- Learn wavelet parameters, bool
            lr_orientation -- Learning rate for the orientation of the scattering parameters
            lr_scattering  -- Learning rate for scattering parameters other than orientation
            skip           -- Whether or not to include skip connections when using learnable filters
            split_filters  -- True to use different wavelets for first and second order scattering     
            subsample      -- Amount of downsampling at each wavelet convolution step
            use_cuda       -- True if using GPU
            plot           -- Plot wavelets when creating the scattering module
        """

        super(sn_ScatteringBase,self).__init__()
        self.J = J
        self.N = N
        self.M = M
        self.channels = channels
        self.max_order = max_order
        self.learnable = learnable
        self.use_cuda = use_cuda 
        self.device = device
        self.initialization = initialization
        self.seed=seed
        self.lr_scattering = lr_scattering
        self.lr_orientation = lr_orientation
        self.skip = skip
        self.split_filters = split_filters
        self.subsample = subsample
        self.M_coefficient = self.M/self.subsample ## Dimensionality of output
        self.N_coefficient = self.N/self.subsample ## fields
        self.epoch = 0
        self.backend = torch_backend.backend

        ## Check for consistent configuration
        if self.learnable==False and self.split_filters:
            if self.split_filters:
                print("Warning: cannot split filters with fixed filters")

        ## Generate smoothing and wavelet filters, and register them as module buffers
        self.phi = create_filters.get_phis(self.M,self.N,self.J)
        self.wavelets, self.params_filters, self.grid = create_filters.create_scatteringExclusive(
            J,N,M,max_order, initialization=self.initialization,seed=seed,
            requires_grad=learnable,use_cuda=self.use_cuda,device=self.device
        )

        ## Determine number of output fields
        if self.max_order==1:
            if self.skip:
                self.n_coefficients=1+len(self.wavelets)
            else:
                self.n_coefficients=len(self.wavelets)
        if self.max_order==2:
            if self.skip and (self.split_filters==False):
                ## Include zeroth and first order fields in forward pass output
                self.n_coefficients=1+len(self.wavelets)+len(self.wavelets)**2
            elif self.skip==False and self.split_filters==False:
                ## Drop skip connections - take only the second order fields
                self.n_coefficients=len(self.wavelets)**2
            elif self.skip and self.split_filters:
                ## Include zeroth and first order fields in forward pass output
                self.n_coefficients=int(1+len(self.wavelets)/2+(len(self.wavelets)/2)**2)
            elif self.skip==False and self.split_filters:
                ## Drop skip connections - take only the second order fields
                self.n_coefficients=int((len(self.wavelets)/2)**2)
            
        self.filterTracker = {'1':[],'2':[],'3':[], 'scale':[], 'angle': []}
        self.filterGradTracker = {'angle': [],'1':[],'2':[],'3':[]}
        if self.J==2 and plot==True:
            self.filters_plots_before = self.getFilterViz()

        self.scatteringTrain = False

    def __str__(self):
        tempL = " L" if self.learnable else "NL"
        tempI = "TF" if self.initialization == 'Tight-Frame' else "R"
        return f"{tempI} {tempL}"

    def train(self,mode=True):
        super().train(mode=mode)
        self.scatteringTrain = True

    def eval(self):
        super().eval()
        if self.scatteringTrain:
            self.updateFilters()
        self.scatteringTrain = False

    def parameters(self):
        """ override parameters to include learning rates """
        if self.learnable:
            yield {'params': [self.params_filters[0]], 'lr': self.lr_orientation, 
                              'maxi_lr':self.lr_orientation , 'weight_decay': 0}
            yield {'params': [ self.params_filters[1],self.params_filters[2],
                               self.params_filters[3]],'lr': self.lr_scattering,
                               'maxi_lr':self.lr_scattering , 'weight_decay': 0}

    def updateFilters(self):
        """if were using learnable scattering, update the filters to reflect 
        the new parameter values obtained from gradient descent"""
        if self.learnable:
            ## Generate new filters
            self.wavelets = create_filters.morlets(self.grid, self.params_filters[0], 
                                              self.params_filters[1], self.params_filters[2], 
                                              self.params_filters[3], device=self.device)
        else:
            pass

    def forward(self, ip):
        """ apply the scattering transform to the input image """

        if (ip.shape[-1] != self.N or ip.shape[-2] != self.M):
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))

        if not torch.is_tensor(ip):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(ip.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not ip.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')

        ## Toggle whether to update filters from backprop, based on model.train
        ## or model.eval settings
        if self.scatteringTrain:
            self.updateFilters()
            
        x = scattering2d.convolve_fields(ip, self.backend, self.J, self.phi, self.wavelets,
                                    self.max_order, self.split_filters,self.subsample)
        x = x[:,:, -self.n_coefficients:,:,:]
        x = x.reshape(x.size(0), self.n_coefficients*self.channels, x.size(3), x.size(4))
        return x

    def countLearnableParams(self):
        """returns the amount of learnable parameters in this model"""
        if not self.learnable:
            return 0

        count = 0
        for t in self.parameters():
            if type(t["params"]) == list:
                for tens in t["params"]: 
                    count += tens.numel()
            else:
                count += t["params"].numel()

        print("Scattering learnable parameters: {}".format(count))
        return count

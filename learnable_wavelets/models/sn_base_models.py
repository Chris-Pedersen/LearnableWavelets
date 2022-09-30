"""Contains all the base pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier

Functions: 
    create_scatteringExclusive -- creates scattering parameters

Exceptions:
    InvalidInitializationException -- Error thrown when an invalid initialization scheme is passed

Classes: 
    sn_Identity -- computes the identity function in forward pass
    sn_HybridModel -- combinations of a scattering and other nn.modules
    sn_ScatteringBase -- a scattering network
"""

import torch
import cv2

import torch.nn as nn

from learnable_wavelets.scattering import scattering2d, create_filters
from learnable_wavelets.models import models_utils
from learnable_wavelets.scattering import torch_backend


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
    """A learnable scattering nn.module 

    parameters:
        learnable -- should the filters be learnable parameters of this model
        use_cuda -- True if we are using cuda
        J -- scale of scattering (always 2 for now)
        N -- height of the input image
        M -- width of the input image
        initilization -- the type of init: ['Tight-Frame' or 'Random']
        seed -- the random seed used to initialize the parameters
    """

    def __init__(self, J, N, M, channels, max_order, initialization, seed, 
                 device, learnable=True, lr_orientation=0.1, lr_scattering=0.1,
                 skip=True, split_filters=False, subsample=1, use_cuda=True,
                 filter_video=False,plot=True):
        """Constructor for the leanable scattering nn.Module
        
        Creates scattering filters and adds them to the nn.parameters if learnable
        
        parameters: 
            J -- scale of scattering (always 2 for now)
            N -- height of the input image
            M -- width of the input image
            max_order -- 1 or 2, haven't implemented higher order than 2 
            initilization -- the type of init: ['Tight-Frame' or 'Random']
            seed -- the random seed used to initialize the parameters
            device -- the device to place weights on
            learnable -- should the filters be learnable parameters of this model
            lr_orientation -- learning rate for the orientation of the scattering parameters
            lr_scattering -- learning rate for scattering parameters other than orientation
            skip -- whether or not to include skip connections when using learnable filters
            split_filters -- split first and second order filters
            subsample -- factor by which to subsample output (1 for no subsampling)
            filter_video -- whether to create filters from 
            use_cuda -- True if using GPU

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
        self.filter_video = filter_video
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

        if self.filter_video:
            self.videoWriters = {}
            self.videoWriters['real'] = cv2.VideoWriter('videos/scatteringFilterProgressionReal{}epochs.avi'.format("--"),
                                              cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
            self.videoWriters['imag'] = cv2.VideoWriter('videos/scatteringFilterProgressionImag{}epochs.avi'.format("--"),
                                              cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
            self.videoWriters['fourier'] = cv2.VideoWriter('videos/scatteringFilterProgressionFourier{}epochs.avi'.format("--"),
                                                 cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)

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


    ### Everything below is concerned with tracking filter evolution
    ### and is not integral to the operation of the scattering model
    def getFilterViz(self):
        """generates plots of the filters for ['fourier','real', 'imag' ] visualizations"""
        wavelets=self.wavelets.contiguous().unsqueeze(3)
        filter_viz = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = models_utils.get_filters_visualization(self.wavelets, self.J, 8, mode=mode) 
            filter_viz[mode] = f  

        return filter_viz

    def getOneFilter(self, count, scale, mode):
        return models_utils.getOneFilter(self.wavelets, count, scale, mode)

    def getAllFilters(self, totalCount, scale, mode):
        return models_utils.getAllFilters(self.wavelets, totalCount, scale, mode)

    def writeVideoFrame(self):
        """Writes frames to the appropriate video writer objects"""
        if self.filter_video:
            for vizType in self.videoWriters.keys():
                temp = cv2.applyColorMap(np.array(self.getAllFilters(totalCount=16, scale=0, mode=vizType),dtype=np.uint8),cv2.COLORMAP_TURBO)
                temp = cv2.putText(temp, "Epoch {}".format(self.epoch),(2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.videoWriters[vizType].write(temp)

    def releaseVideoWriters(self):
        if self.filter_video:
            for vizType in self.videoWriters.keys():
                self.videoWriters[vizType].release()

    def setEpoch(self, epoch):
        self.epoch = epoch

    def checkParamDistance(self):
        """Method to checking the minimal distance between initialized filters and learned ones
        
        Euclidean distances are calculated between each filter for parameters other than orientations
        for orientations, we calculate the arc between both points on the unit circle. Then, the sum of
        these two distances becomes the distance between two filters. Finally, we use munkre's assignment 
        algorithm to compute the optimal match (I.E. the one that minizes total distance)        

        return: 
            minimal distance
        """
        tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in self.params_filters[1:]],dim=1)
        tempParamsAngle = self.params_filters[0] % (2 * np.pi)
        self.params_history.append({'params':tempParamsGrouped,'angle':tempParamsAngle})

        return models_utils.compareParams(
            params1=tempParamsGrouped,
            angles1=tempParamsAngle, 
            params2=self.compared_params_grouped,
            angles2=self.compared_params_angle,
            device=self.device
        )

    def compareParamsVisualization(self):
        """visualize the matched filters"""
        tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in self.params_filters[1:]],dim=1)
        tempParamsAngle = self.params_filters[0] % (2 * np.pi)
        self.params_history.append({'params':tempParamsGrouped,'angle':tempParamsAngle})

        return models_utils.compareParamsVisualization(
            params1=tempParamsGrouped,
            angles1=tempParamsAngle, 
            params2=self.compared_params_grouped,
            angles2=self.compared_params_angle,
            device=self.device
        )

    def saveFilterValues(self,scatteringActive):
        try:
            if scatteringActive:
                orientations1 = self.params_filters[0].detach().clone()
                self.filterTracker['1'].append(self.params_filters[1].detach().clone())
                self.filterTracker['2'].append(self.params_filters[2].detach().clone()) 
                self.filterTracker['3'].append(self.params_filters[3].detach().clone()) 
                scale = torch.mul(self.params_filters[1].detach().clone(), self.params_filters[2].detach().clone())
                self.filterTracker['scale'].append(scale) 
                self.filterTracker['angle'].append(orientations1) 

            else:
                self.filterGradTracker['angle'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['3'].append(torch.zeros(self.params_filters[1].shape[0]))
        except Exception:
            pass

    def saveFilterGrads(self,scatteringActive):
        try:
            if scatteringActive:
                self.filterGradTracker['angle'].append(self.params_filters[0].grad.clone()) 
                self.filterGradTracker['1'].append(self.params_filters[1].grad.clone()) 
                self.filterGradTracker['2'].append(self.params_filters[2].grad.clone()) 
                self.filterGradTracker['3'].append(self.params_filters[3].grad.clone()) 
            else:
                self.filterGradTracker['angle'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['1'].append(torch.zeros(self.params_filters[1].shape[0])) 
                self.filterGradTracker['2'].append(torch.zeros(self.params_filters[1].shape[0]))
                self.filterGradTracker['3'].append(torch.zeros(self.params_filters[1].shape[0]))
        except Exception:
            pass

    def plotFilterGrads(self):
        """plots the graph of the filter gradients"""
        filterNum = self.params_filters[1].shape[0]
        col = 8
        row = int(filterNum/col)
        size = (80, 10*row,)

        f, axarr = plt.subplots(row, col, figsize=size) # create plots

        for x in range(filterNum):#iterate over all the filters
            temp = {
                'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterGradTracker['angle']],
                'xis': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['1']],
                'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['2']],
                'slant': [float(filters[x].cpu().numpy())  for filters in self.filterGradTracker['3']],
            }

            axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='orientation1')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
            axarr[int(x/col),x%col].plot([x  for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
            axarr[int(x/col),x%col].legend()

        return f
    
    def plotFilterValues(self):
        """plots the graph of the filter values"""
        filterNum = self.params_filters[1].shape[0]
        col = 8
        row = int(filterNum/col)
        size = (80, 10*row,)

        f, axarr = plt.subplots(row, col, figsize=size) # create plots

        for x in range(filterNum):#iterate over all the filters
            #axarr[int(x/col),x%col].axis('off')
            temp = {
                'orientation1': [float(filters[x].cpu().numpy()) for filters in self.filterTracker['angle']],
                'xis': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['1']],
                'sigmas': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['2']],
                'slant': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['3']],
                'scale': [float(filters[x].cpu().numpy())  for filters in self.filterTracker['scale']],
            }

            axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
            axarr[int(x/col),x%col].plot([x for x in range(len(temp['scale']))],temp['scale'],color='black', label='scale')
            axarr[int(x/col),x%col].legend()

        return f
        
    def plotParameterValues(self):
        size = (10, 10)
        f, axarr = plt.subplots(2, 2, figsize=size) # create plots
        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        label = ['theta','xis','sigma','slant']

        for idx,param in enumerate(['angle',"1",'2','3']):#iterate over all the parameters
            for idx2,filter in enumerate(torch.stack(self.filterTracker[param]).T):
                filter = filter.cpu().numpy()
                # if param == 'angle':
                #     filter = filter%(2*np.pi)
                axarr[int(idx/2),idx%2].plot([x for x in range(len(filter))],filter)#, label=idx2)
            # axarr[int(idx/2),idx%2].legend()
            axarr[int(idx/2),idx%2].set_title(label[idx], fontsize=16)
            axarr[int(idx/2),idx%2].set_xlabel('Epoch', fontsize=12) # Or ITERATION to be more precise
            axarr[int(idx/2),idx%2].set_ylabel('Value', fontsize=12)
        
        return f


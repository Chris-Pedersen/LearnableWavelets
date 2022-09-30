import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time, sys, os
import matplotlib.pyplot as plt

# my modules
from learnable_wavelets.models.models_factory import baseModelFactory, topModelFactory
from learnable_wavelets.models.sn_hybrid_models import sn_HybridModel
from learnable_wavelets.models.camels_models import model_o3_err
from learnable_wavelets.camels.camels_dataset import *
from learnable_wavelets.utils.test_model import test_model

""" Base script to test a scattering network on a CAMELs dataset """

## Check if CUDA available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
    use_cuda=True
else:
    print('CUDA Not Available')
    device = torch.device('cpu')
    use_cuda=False



cudnn.benchmark = True      #May train faster but cost more memory


#######################################################################################################
#######################################################################################################
############################## Set up training params #################################################
#######################################################################################################
#######################################################################################################
## camels path
camels_path=os.environ['CAMELS_PATH']

# data parameters
fmaps      = ['maps_Mcdm.npy'] #tuple containing the maps with the different fields to consider
fmaps_norm = [None] #if you want to normalize the maps according to the properties of some data set, put that data set here (This is mostly used when training on IllustrisTNG and testing on SIMBA, or vicerversa)
fparams    = camels_path+"/params_IllustrisTNG.txt"
seed       = 1   #random seed to split maps among training, validation and testing
splits     = 6   #number of maps per simulation

# training parameters
channels        = 1                #we only consider here 1 field
params          = [0,1,2,3,4,5]    #0(Omega_m) 1(sigma_8) 2(A_SN1) 3 (A_AGN1) 4(A_SN2) 5(A_AGN2). The code will be trained to predict all these parameters.
g               = params           #g will contain the mean of the posterior
h               = [6+i for i in g] #h will contain the variance of the posterior
rot_flip_in_mem = False            #whether rotations and flipings are kept in memory. True will make the code faster but consumes more RAM memory.

# optimizer parameters
beta1 = 0.5
beta2 = 0.999

# hyperparameters
batch_size = 128
lr         = 1e-3
wd         = 0.0005  #value of weight decay
dr         = 0.2    #dropout value for fully connected layers
hidden     = 5      #this determines the number of channels in the CNNs; integer larger than 1
epochs     = 100    #number of epochs to train the network

# output files names
floss  = 'loss.txt'   #file with the training and validation losses for each epoch
fmodel = 'weights.pt' #file containing the weights of the best-model
#######################################################################################################
#######################################################################################################


fmaps2 = camels_path+"/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"
maps  = np.load(fmaps2)
print('Shape of the maps:',maps.shape)
# define the array that will contain the indexes of the maps
indexes = np.zeros(1000*splits, dtype=np.int32)

# do a loop over all maps and choose the ones we want
count = 0
for i in range(15000):
    if i%15 in np.arange(splits):  
      indexes[count] = i
      count += 1
print('Selected %d maps out of 15000'%count)

# save these maps to a new file
maps = maps[indexes]
np.save('maps_Mcdm.npy', maps)
del maps

# This routine returns the data loader need to train the network
def create_dataset_multifield(mode, seed, fmaps, fparams, batch_size, splits, fmaps_norm,
                              rot_flip_in_mem=rot_flip_in_mem, shuffle=True, verbose=False):

    # whether rotations and flippings are kept in memory
    if rot_flip_in_mem:
        data_set = make_dataset_multifield(mode, seed, fmaps, fparams, splits, fmaps_norm, verbose)
    else:
        data_set = make_dataset_multifield2(mode, seed, fmaps, fparams, splits, fmaps_norm, verbose)

    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader


# get test set
print('\nPreparing validation set')
test_loader = create_dataset_multifield('test', seed, fmaps, fparams, batch_size, splits, fmaps_norm, 
                                         rot_flip_in_mem=True,  verbose=True)    


## In this script we load the weights for a given trial from the CAMELs MFD work
trial_number = 1
fweights   = '/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/2D_maps/inference/weights/weights_IllustrisTNG_Mcdm_%d_all_steps_500_500_o3.pt' % trial_number
fdatabase  = 'sqlite:////mnt/home/cpedersen/Data/CAMELS_test/databases/IllustrisTNG_o3_Mcdm_all_steps_500_500_o3.db'
study_name = 'wd_dr_hidden_lr_o3' 

study = optuna.load_study(study_name=study_name, storage=fdatabase)

trial = study.trials[trial_number]
print("Trial number:  number {}".format(trial.number))
print("Loss:          %.5e"%trial.value)
print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

model = model_o3_err(trial.params['hidden'], trial.params['dr'], 1)
model = nn.DataParallel(model)
model.to(device=device)
network_total_params = sum(p.numel() for p in model.parameters())
print('total number of parameters in the model = %d'%network_total_params)

if os.path.exists(fweights):  
    model.load_state_dict(torch.load(fweights, map_location=torch.device(device)))
    print('Weights loaded')
else:
    raise Exception('file with weights not found!!!')

test_model(model,test_loader,device)


import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time, sys, os
import matplotlib.pyplot as plt
import wandb

# my modules
from learnable_wavelets.models.models_factory import baseModelFactory, topModelFactory
from learnable_wavelets.models.sn_hybrid_models import sn_HybridModel
from learnable_wavelets.models.camels_models import model_o3_err
from learnable_wavelets.camels.camels_dataset import *

""" Script to test the a model on a single map """
epochs=2000
lr=1e-5
batch_size=1
model_type="sn" ## "sn" or "camels" for now
# hyperparameters
wd         = 0.0001  #value of weight decay
dr         = 0.00    #dropout value for fully connected layers
hidden     = 5      #this determines the number of channels in the CNNs; integer larger than 1


config = {"learning rate": lr,
                 "epochs": epochs,
                 "batch size": batch_size,
                 "network": model_type,
                 "weight decay": wd,
                 "dropout": dr}

## Initialise wandb
wandb.login()
wandb.init(project="my-test-project", entity="chris-pedersen",config=config)


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
## model type
model_type="sn" ## "sn" or "camels" for now

## camels path
camels_path=os.environ['CAMELS_PATH']

# data parameters
fmaps      = ['maps_Mcdm_single.npy'] #tuple containing the maps with the different fields to consider
fmaps_norm = [None] #if you want to normalize the maps according to the properties of some data set, put that data set here (This is mostly used when training on IllustrisTNG and testing on SIMBA, or vicerversa)
fparams    = camels_path+"/params_IllustrisTNG.txt"
seed       = 1   #random seed to split maps among training, validation and testing
splits     = 1   #number of maps per simulation

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
wd         = 0.0005  #value of weight decay
dr         = 0.2    #dropout value for fully connected layers
hidden     = 5      #this determines the number of channels in the CNNs; integer larger than 1

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
np.save('maps_Mcdm_single.npy', maps)
del maps

# get training set
print('\nPreparing training set')
train_loader = create_dataset_multifield('train', seed, fmaps, fparams, batch_size, splits, fmaps_norm, 
                                         rot_flip_in_mem=rot_flip_in_mem, verbose=True)

# get validation set
print('\nPreparing validation set')
valid_loader = create_dataset_multifield('valid', seed, fmaps, fparams, batch_size, splits, fmaps_norm, 
                                         rot_flip_in_mem=True,  verbose=True)    


if model_type=="sn":
    ## First create a scattering network object
    scatteringBase = baseModelFactory( #creat scattering base model
        architecture='scattering',
        J=2,
        N=256,
        M=256,
        second_order=True,
        initialization="Random",
        seed=123,
        learnable=True,
        lr_orientation=0.1,
        lr_scattering=0.1,
        filter_video=False,
        device=device,
        use_cuda=use_cuda
    )

    ## Now create a network to follow the scattering layers
    ## can be MLP, linear, or cnn at the moment
    ## (as in https://github.com/bentherien/ParametricScatteringNetworks/ )
    top = topModelFactory( #create cnn, mlp, linearlayer, or other
        base=scatteringBase,
        architecture="mlp",
        num_classes=12,
        width=8,
        use_cuda=use_cuda
    )

    ## Merge these into a hybrid model
    hybridModel = sn_HybridModel(scatteringBase=scatteringBase, top=top, use_cuda=use_cuda)
    model=hybridModel
    print("scattering layer + cnn set up")
elif model_type=="camels":
    model = model_o3_err(hidden, dr, channels)
    print("camels cnn model set up")
else:
    print("model type %s not recognised" % model_type)
model.to(device=device)

## wandb
wandb.watch(model, log_freq=10)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=10)

print("Train on a single map")
## Single data batch
x, y=next(iter(train_loader))

print("Training data shape: ",x.shape)

# do a loop over all epochs
start = time.time()
if model_type=="camels" and batch_size==1:
    model.eval()
    print("Dropping batchnorm for camels cnn")
else:
    model.train()

for epoch in range(epochs):
    # do training
    train_loss1, train_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    train_loss, points = 0.0, 0
    #for x, y in train_loader:
    bs   = x.shape[0]         #batch size
    x    = x.to(device)       #maps
    y    = y.to(device)[:,g]  #parameters
    p    = model(x)           #NN output
    y_NN = p[:,g]             #posterior mean
    e_NN = p[:,h]             #posterior std
    #print("Map=",x)
    #print("Params=",y)
    #print("Prediction=",y_NN)
    loss1 = torch.mean((y_NN - y)**2,                axis=0)
    loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
    loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
    train_loss1 += loss1*bs
    train_loss2 += loss2*bs
    points      += bs
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

        #if points>18000:  break
    train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
    train_loss = torch.mean(train_loss).item()

    # verbose
    print('%03d %.3e %.3e '%(epoch, train_loss, train_loss), end='')
    print("")

    if epoch % 10 == 0:
        wandb.log({"loss": train_loss})

stop = time.time()
print('Time take (h):', "{:.4f}".format((stop-start)/3600.0))

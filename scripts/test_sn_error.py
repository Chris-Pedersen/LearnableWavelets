import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time, sys, os
import wandb

# my modules
from sn_camels.models.models_factory import baseModelFactory, topModelFactory
from sn_camels.models.sn_hybrid_models import sn_HybridModel
from sn_camels.camels.camels_dataset import *
from sn_camels.models.camels_models import get_architecture
from sn_camels.utils.test_model import test_model

""" Base script to test a scattering network on a CAMELs dataset """

epochs=100
lr=1e-3
batch_size=32
project_name="linear_layer"
error=True # Predict errors?
model_type="sn" ## "sn" or "camels" for now
# hyperparameters
wd         = 0.0005  #value of weight decay
dr         = 0.2    #dropout value for fully connected layers
hidden     = 5      #this determines the number of channels in the CNNs; integer larger than 1

seed       = 1   #random seed to split maps among training, validation and testing
splits     = 1   #number of maps per simulation

config = {"learning rate": lr,
                 "epochs": epochs,
                 "batch size": batch_size,
                 "network": model_type,
                 "dropout": dr,
                 "error": error,
                 "splits":splits}

## Initialise wandb
wandb.login()
wandb.init(project="%s" % project_name, entity="chris-pedersen",config=config)

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

############################## Set up training params #################################################
## camels path
camels_path=os.environ['CAMELS_PATH']

# data parameters
fmaps      = ['maps_Mcdm.npy'] #tuple containing the maps with the different fields to consider
fmaps_norm = [None] #if you want to normalize the maps according to the properties of some data set, put that data set here (This is mostly used when training on IllustrisTNG and testing on SIMBA, or vicerversa)
fparams    = camels_path+"/params_IllustrisTNG.txt"

# training parameters
channels        = 1                #we only consider here 1 field
params          = [0,1,2,3,4,5]    #0(Omega_m) 1(sigma_8) 2(A_SN1) 3 (A_AGN1) 4(A_SN2) 5(A_AGN2). The code will be trained to predict all these parameters.
g               = params           #g will contain the mean of the posterior
h               = [6+i for i in g] #h will contain the variance of the posterior
rot_flip_in_mem = False            #whether rotations and flipings are kept in memory. True will make the code faster but consumes more RAM memory.

## Set number of classes for scattering network to output
if error==True:
    sn_classes=12
else:
    sn_classes=6

# optimizer parameters
beta1 = 0.5
beta2 = 0.999
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

# get training set
print('\nPreparing training set')
train_loader = create_dataset_multifield('train', seed, fmaps, fparams, batch_size, splits, fmaps_norm, 
                                         rot_flip_in_mem=rot_flip_in_mem, verbose=True)

# get validation set
print('\nPreparing validation set')
valid_loader = create_dataset_multifield('valid', seed, fmaps, fparams, batch_size, splits, fmaps_norm, 
                                         rot_flip_in_mem=rot_flip_in_mem,  verbose=True)    

# get test set
print('\nPreparing test set')
test_loader = create_dataset_multifield('test', seed, fmaps, fparams, batch_size, splits, fmaps_norm,
                                         rot_flip_in_mem=rot_flip_in_mem,  verbose=True)

num_train_maps=len(train_loader.dataset.x)
wandb.config.update({"no. training maps": num_train_maps})

if model_type=="sn":
    ## First create a scattering network object
    scatteringBase = baseModelFactory( #creat scattering base model
        architecture='scattering',
        J=2,
        N=256,
        M=256,
        max_order=2,
        initialization="Tight-Frame",
        seed=123,
        learnable=True,
        lr_orientation=0.005,
        lr_scattering=0.005,
        skip=False,
        split_filters=False,
        filter_video=False,
        subsample=4,
        device=device,
        use_cuda=use_cuda
    )

    ## Now create a network to follow the scattering layers
    ## can be MLP, linear, or cnn at the moment
    ## (as in https://github.com/bentherien/ParametricScatteringNetworks/ )
    top = topModelFactory( #create cnn, mlp, linearlayer, or other
        base=scatteringBase,
        architecture="linear_layer",
        num_classes=sn_classes,
        width=5,
        average=True,
        use_cuda=use_cuda
    )

    ## Merge these into a hybrid model
    hybridModel = sn_HybridModel(scatteringBase=scatteringBase, top=top, use_cuda=use_cuda)
    model=hybridModel
    wandb.config.update({"learnable":scatteringBase.learnable,
                         "learnable_parameters":model.countLearnableParams(),
                         "max_order":scatteringBase.max_order,
                         "skip":scatteringBase.skip,
                         "split_filters":scatteringBase.split_filters,
                         "subsample":scatteringBase.subsample,
                         "scattering_output_dims":scatteringBase.M_coefficient,
                         "n_coefficients":scatteringBase.n_coefficients,
                         "top_model":top.arch,
                         "spatial_average":top.average
                         })
    print("scattering layer + cnn set up")
else:
    print("setting up model %s" % model_type)
    model = get_architecture(model_type,hidden,dr,channels)
    wandb.config.update({"learnable_parameters":sum(p.numel() for p in model.parameters())})
model.to(device=device)

# wandb
wandb.watch(model, log_freq=1)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=10)

# do a loop over all epochs
start = time.time()
for epoch in range(epochs):
    log_dic={}
    if model_type=="sn":
        wave_params=hybridModel.scatteringBase.params_filters
        orientations=wave_params[0].cpu().detach().numpy()
        xis=wave_params[1].cpu().detach().numpy()
        sigmas=wave_params[2].cpu().detach().numpy()
        slants=wave_params[3].cpu().detach().numpy()
            log_dic["slant_%d" % aa]=slants[aa]

    # do training
    train_loss1, train_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    train_loss, points = 0.0, 0
    model.train()
    for x, y in train_loader:
        bs   = x.shape[0]         #batch size
        x    = x.to(device)       #maps
        y    = y.to(device)[:,g]  #parameters
        p    = model(x)           #NN output
        y_NN = p[:,g]             #posterior mean
        loss1 = torch.mean((y_NN - y)**2,                axis=0)
        if error==True:
            e_NN = p[:,h]         #posterior std
            loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
            loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
            train_loss2 += loss2*bs
        else:
            loss = torch.mean(torch.log(loss1))
        train_loss1 += loss1*bs
        points      += bs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = torch.log(train_loss1/points) 
    if error==True:
        train_loss+=torch.log(train_loss2/points)
    train_loss = torch.mean(train_loss).item()

    # do validation: cosmo alone & all params
    valid_loss1, valid_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    valid_loss, points = 0.0, 0
    model.eval()
    for x, y in valid_loader:
        with torch.no_grad():
            bs    = x.shape[0]         #batch size
            x     = x.to(device)       #maps
            y     = y.to(device)[:,g]  #parameters
            p     = model(x)           #NN output
            y_NN  = p[:,g]             #posterior mean
            loss1 = torch.mean((y_NN - y)**2,                axis=0)
            if error==True:    
                e_NN  = p[:,h]         #posterior std
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                valid_loss2 += loss2*bs
            valid_loss1 += loss1*bs
            points     += bs

    
    valid_loss = torch.log(valid_loss1/points) 
    if error==True:
        valid_loss+=torch.log(valid_loss2/points)
    valid_loss = torch.mean(valid_loss).item()

    scheduler.step(valid_loss)
    log_dic["training_loss"]=train_loss
    log_dic["valid_loss"]=valid_loss
    wandb.log(log_dic)
            
    # verbose
    print('%03d %.3e %.3e '%(epoch, train_loss, valid_loss), end='')
    print("")

stop = time.time()
print('Time take (h):', "{:.4f}".format((stop-start)/3600.0))

#test_model(model,test_loader,device)

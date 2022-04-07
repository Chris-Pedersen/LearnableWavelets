import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import time, sys, os
import matplotlib.pyplot as plt

# my modules
from sn_camels.models.models_factory import baseModelFactory, topModelFactory
from sn_camels.models.sn_hybrid_models import sn_HybridModel
from sn_camels.models.camels_models import get_architecture
from sn_camels.models.camels_models import model_o3_err
from sn_camels.camels.camels_dataset import *


import optuna
import wandb

## Everything done within an Objective class now for optuna
class Objective(object):
    def __init__(self, device, seed, fmaps, fmaps_norm, fparams, batch_size, splits,
                      arch, hidden, beta1, beta2, epochs, monopole, name,
                      num_workers, params, rot_flip_in_mem, smoothing):
        self.device          = device
        self.seed            = seed
        self.fmaps           = fmaps
        self.fmaps_norm      = fmaps_norm
        self.fparams         = fparams
        self.batch_size      = batch_size
        self.splits          = splits
        self.arch            = arch
        self.hidden          = hidden
        self.beta1           = beta1
        self.beta2           = beta2
        self.epochs          = epochs
        self.monopole        = monopole
        self.name            = name
        self.num_workers     = num_workers
        self.params          = params
        self.rot_flip_in_mem = rot_flip_in_mem
        self.smoothing       = smoothing
        print("Done init")

    def __call__(self,trial):
        ## number of fields - hardcoded to 1 for now
        channels  = 1

        # tuple with the indexes of the parameters to train
        g = self.params      #posterior mean
        h = [6+i for i in g] #posterior variance

        print("Suggesting trial")
        # get the value of the hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd     = trial.suggest_float("wd", 1e-8, 1e-1, log=True)
        dr     = trial.suggest_float("dr", 0.0,  0.9)
        print("Suggested trial")

        print('\nTrial number: {}'.format(trial.number))
        print('lr: {}'.format(lr))
        print('wd: {}'.format(wd))
        print('dr: {}'.format(dr))
        
        config = {"learning rate": lr,
                 "epochs": self.epochs,
                 "batch size": self.batch_size,
                 "network": self.arch,
                 "dropout": dr,
                 "splits": self.splits}
    

        ## Initialise wandb
        wandb.login()
        wandb.init(project="%s" % name, entity="chris-pedersen",config=config)

        ### LOAD DATA
        ## camels path
        camels_path="/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/2D_maps/data/"

        # get training set
        print('\nPreparing training set')
        train_loader = create_dataset_multifield('train', self.seed, self.fmaps, self.fparams, self.batch_size, self.splits, self.fmaps_norm, 
                                                rot_flip_in_mem=self.rot_flip_in_mem, verbose=True)

        # get validation set
        print('\nPreparing validation set')
        valid_loader = create_dataset_multifield('valid', self.seed, self.fmaps, self.fparams, self.batch_size, self.splits, self.fmaps_norm, 
                                                rot_flip_in_mem=self.rot_flip_in_mem,  verbose=True)

        num_train_maps=len(train_loader.dataset.x)
        wandb.config.update({"no. training maps": num_train_maps})
        print("setting up model %s" % self.arch)
        model = get_architecture(self.arch,self.hidden,dr,channels)
        wandb.config.update({"learnable_parameters":sum(p.numel() for p in model.parameters())})
        model.to(device=device)

        # wandb
        wandb.watch(model, log_freq=10)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=10)

        print('Computing initial validation loss')
        model.eval()
        valid_loss1, valid_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
        min_valid_loss, points = 0.0, 0
        for x, y in valid_loader:
            with torch.no_grad():
                bs   = x.shape[0]                #batch size
                x    = x.to(device=device)       #maps
                y    = y.to(device=device)[:,g]  #parameters
                p    = model(x)                  #NN output
                y_NN = p[:,g]                    #posterior mean
                e_NN = p[:,h]                    #posterior std
                loss1 = torch.mean((y_NN - y)**2,                axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                valid_loss1 += loss1*bs
                valid_loss2 += loss2*bs
                points += bs
        min_valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
        min_valid_loss = torch.mean(min_valid_loss).item()
        print('Initial valid loss = %.3e'%min_valid_loss)

        # do a loop over all epochs
        start = time.time()
        for epoch in range(epochs):

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
                e_NN = p[:,h]             #posterior std
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
                    e_NN  = p[:,h]             #posterior std
                    loss1 = torch.mean((y_NN - y)**2,                axis=0)
                    loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                    loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                    valid_loss1 += loss1*bs
                    valid_loss2 += loss2*bs
                    points     += bs
            valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
            valid_loss = torch.mean(valid_loss).item()

            scheduler.step(valid_loss)
            wandb.log({"training loss": train_loss, "validation loss": valid_loss})


            # verbose
            print('%03d %.3e %.3e '%(epoch, train_loss, valid_loss), end='')
            print("")

        stop = time.time()
        print('Time take (h):', "{:.4f}".format((stop-start)/3600.0))

        wandb.finish()

        return min_valid_loss

##################################### INPUT ##########################################
# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')
cudnn.benchmark = True      #May train faster but cost more memory

# architecture parameters
beta1 = 0.5
beta2 = 0.999

## My stuff
## camels path
name       ="optuna_camels_deep"
camels_path="/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/2D_maps/data/"
fparams    = camels_path+"/params_IllustrisTNG.txt"
fmaps      = ['maps_Mcdm.npy']
fmaps_norm = [None]
splits     = 1
seed       = 123
params     = [0,1,2,3,4,5] #0(Om) 1(s8) 2(A_SN1) 3 (A_AGN1) 4(A_SN2) 5(A_AGN2)
monopole        = True  #keep the monopole of the maps (True) or remove it (False)
rot_flip_in_mem = False  #whether rotations and flipings are kept in memory
smoothing  = 0  ## Smooth the maps with a Gaussian filter? 0 for no
arch = "o3_err" ## Which model architecture to use    


fmaps2 = camels_path+"/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"
maps  = np.load(fmaps2)
print('Shape of the maps:',maps.shape)
# define the array that will contain the indexes of the maps
indexes = np.zeros(1000*splits, dtype=np.int32)

# do a loop over all maps and choose the ones we want
count = 0
for i in range(5000):
    if i%15 in np.arange(splits):  
      indexes[count] = i
      count += 1
print('Selected %d maps out of 15000'%count)

# save these maps to a new file
maps = maps[indexes]
np.save('maps_Mcdm.npy', maps)
del maps

## training parameters
batch_size  = 32
hidden      = 5
epochs      = 100
num_workers = 1    #number of workers to load data

## Optuna params
study_name = "optuna/"+name  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
n_trials=35

# train networks with bayesian optimization
objective = Objective(device, seed, fmaps, fmaps_norm, fparams, batch_size, splits,
                      arch, hidden, beta1, beta2, epochs, monopole, name,
                    num_workers, params, rot_flip_in_mem, smoothing)
sampler = optuna.samplers.TPESampler(n_startup_trials=20)
study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage_name,
                            load_if_exists=True)
study.optimize(objective, n_trials, gc_after_trial=False)

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
from sn_camels.models.camels_models import model_o3_err
from sn_camels.camels.camels_dataset import *

import optuna
import wandb

## Everything done within an Objective class now for optuna
class Objective(object):
    def __init__(self, device, seed, fmaps, fmaps_norm, fparams, batch_size, splits,
                      arch, min_lr, beta1, beta2, epochs, monopole,
                      num_workers, params, rot_flip_in_mem, smoothing):
        self.device          = device
        self.seed            = seed
        self.fmaps          = fmaps
        self.fmaps_norm     = fmaps_norm
        self.fparams        = fparams
        self.batch_size      = batch_size
        self.splits          = splits
        self.arch            = arch
        self.min_lr          = min_lr
        self.beta1           = beta1
        self.beta2           = beta2
        self.epochs          = epochs
        self.monopole        = monopole
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
        #hidden = trial.suggest_int("hidden", 6, 12)
        print("Suggested trial")
        ## Store hyperparams in a config for wandb
        config = {"learning rate": lr,
                 "epochs": self.epochs,
                 "batch size": self.batch_size,
                 "network": self.arch,
                 "weight decay": wd,
                 "dropout": dr,
                 "splits":self.splits}
        
        print('\nTrial number: {}'.format(trial.number))
        print('lr: {}'.format(lr))
        print('wd: {}'.format(wd))
        print('dr: {}'.format(dr))
        #print('hidden: {}'.format(hidden))

        ## Initialise wandb
        wandb.login()
        wandb.init(project="optuna-test", entity="chris-pedersen",config=config)

        ### LOAD DATA
        ## camels path
        camels_path=os.environ['CAMELS_PATH']

        # get training set
        print('\nPreparing training set')
        train_loader = create_dataset_multifield('train', self.seed, self.fmaps, self.fparams, self.batch_size, self.splits, self.fmaps_norm, 
                                                rot_flip_in_mem=self.rot_flip_in_mem, verbose=True)

        # get validation set
        print('\nPreparing validation set')
        valid_loader = create_dataset_multifield('valid', self.seed, self.fmaps, self.fparams, self.batch_size, self.splits, self.fmaps_norm, 
                                                rot_flip_in_mem=self.rot_flip_in_mem,  verbose=True)

        num_train_maps=train_loader.dataset.x.size
        wandb.config.update({"no. training maps": num_train_maps})

        if self.arch=="sn":
            ## First create a scattering network object
            scatteringBase = baseModelFactory( #creat scattering base model
                architecture='scattering',
                J=2,
                N=256,
                M=256,
                second_order=True,
                initialization="Random",
                seed=123,
                learnable=False,
                lr_orientation=0.1,
                lr_scattering=0.1,
                filter_video=False,
                device=device,
                use_cuda=True
            )

            ## Now create a network to follow the scattering layers
            ## can be MLP, linear, or cnn at the moment
            ## (as in https://github.com/bentherien/ParametricScatteringNetworks/ )
            top = topModelFactory( #create cnn, mlp, linearlayer, or other
                base=scatteringBase,
                architecture="cnn",
                num_classes=12,
                width=8,
                use_cuda=True
            )

            ## Merge these into a hybrid model
            hybridModel = sn_HybridModel(scatteringBase=scatteringBase, top=top, use_cuda=use_cuda)
            model=hybridModel
            print("scattering layer + cnn set up")
        elif self.arch=="camels":
            model = model_o3_err(self.hidden, dr, channels)
            print("camels cnn model set up")
        else:
            print("model type %s not recognised" % self.arch)
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

            
            wandb.log({"training loss": train_loss})

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
            wandb.log({"validation loss": valid_loss})


            # verbose
            print('%03d %.3e %.3e '%(epoch, train_loss, valid_loss), end='')

        stop = time.time()
        print('Time take (h):', "{:.4f}".format((stop-start)/3600.0))

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
camels_path=os.environ['CAMELS_PATH']
fparams    = camels_path+"/params_IllustrisTNG.txt"
fmaps      = ['maps_Mcdm.npy']
fmaps_norm = [None]
splits     = 6
seed       = 123
params     = [0,1,2,3,4,5] #0(Om) 1(s8) 2(A_SN1) 3 (A_AGN1) 4(A_SN2) 5(A_AGN2)
monopole        = True  #keep the monopole of the maps (True) or remove it (False)
rot_flip_in_mem = True  #whether rotations and flipings are kept in memory
smoothing  = 0  ## Smooth the maps with a Gaussian filter? 0 for no
arch = "camels" ## Which model architecture to use    

## training parameters
batch_size  = 128
min_lr      = 1e-9
epochs      = 200
num_workers = 1    #number of workers to load data

## Optuna params
study_name = "optuna/example-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
n_trials=1

# train networks with bayesian optimization
objective = Objective(device, seed, fmaps, fmaps_norm, fparams, batch_size, splits,
                      arch, min_lr, beta1, beta2, epochs, monopole, 
                    num_workers, params, rot_flip_in_mem, smoothing)
sampler = optuna.samplers.TPESampler(n_startup_trials=20)
study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage_name,
                            load_if_exists=True)
study.optimize(objective, n_trials, gc_after_trial=False)

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

## Everything done within an Objective class now for optuna
class Objective(object):
    def __init__(self,device,seed,f_maps,fparams,splits,channels,g,h,rot_flip_in_mem,
                    beta1,beta2,epochs,)
    

    def __call__(self,trial):
        # get the number of channels in the maps
        channels  = len(self.fields)

        # tuple with the indexes of the parameters to train
        g = self.params      #posterior mean
        h = [6+i for i in g] #posterior variance

        # get the value of the hyperparameters
        max_lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd     = trial.suggest_float("wd", 1e-8, 1e-1, log=True)
        dr     = trial.suggest_float("dr", 0.0,  0.9)
        #hidden = trial.suggest_int("hidden", 6, 12)

        ## Store hyperparams in a config for wandb
        config = {"learning rate": lr,
                 "epochs": epochs,
                 "batch size": batch_size,
                 "network": model_type,
                 "weight decay": wd,
                 "dropout": dr,
                 "splits":splits}

        ## Initialise wandb
        wandb.login()
        wandb.init(project="my-test-project", entity="chris-pedersen",config=config)
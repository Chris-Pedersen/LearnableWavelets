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
from sn_camels.models.camels_models import get_architecture
from sn_camels.camels.camels_dataset import *

import optuna
import wandb

## Everything done within an Objective class now for optuna
class Objective(object):
    def __init__(self, device, seed, fmaps, fmaps_norm, fparams, batch_size, splits,
                       arch, features, beta1, beta2, epochs, monopole,
                       num_workers, params, rot_flip_in_mem, smoothing, name):
        self.device          = device
        self.seed            = seed
        self.fmaps           = fmaps
        self.fmaps_norm      = fmaps_norm
        self.fparams         = fparams
        self.batch_size      = batch_size
        self.splits          = splits
        self.arch            = arch
        self.features        = features
        self.beta1           = beta1
        self.beta2           = beta2
        self.epochs          = epochs
        self.monopole        = monopole
        self.num_workers     = num_workers
        self.params          = params
        self.rot_flip_in_mem = rot_flip_in_mem
        self.smoothing       = smoothing
        self.name            = name
        
    def __call__(self,trial):

        ## CNN values
        print("Suggesting trial")
        # get the value of the hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd     = trial.suggest_float("wd", 1e-8, 1e-1, log=True)
        dr     = trial.suggest_float("dr", 0.0,  0.6)
        hidden = trial.suggest_int("hidden", 3,  10)
        print("Suggested trial")

        print('\nTrial number: {}'.format(trial.number))
        print('lr: {}'.format(lr))
        print('wd: {}'.format(wd))
        print('dr: {}'.format(dr))
        print('hidden: {}'.format(hidden))
        lr_sn="na"
        #dr="na"
        
        ''''
        ## SN values
        print("Suggesting trial")
        # get the value of the hyperparameters
        lr     = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        lr_sn  = trial.suggest_float("lr_sn", 1e-4, 5e-2, log=True)
        wd     = trial.suggest_float("wd", 1e-6, 1e-1, log=True)
        hidden = trial.suggest_int("hidden", 1,  8)
        print("Suggested trial")

        print('\nTrial number: {}'.format(trial.number))
        print('lr: {}'.format(lr))
        print('lr_sn: {}'.format(lr_sn))
        print('wd: {}'.format(wd))
        print('hidden: {}'.format(hidden))
        '''

        
        # training parameters
        channels        = len(fmaps)
        ## Set up params
        if self.features==2:
            g=[0,1]
        if self.features==4:
            g=[0,1]
            h=[2,3]
        elif self.features==6:
            g=[0,1,2,3,4,5]
        elif self.features==12:
            g=[0,1,2,3,4,5]
            h=[6,7,8,9,10,11]

        rot_flip_in_mem = False            #whether rotations and flipings are kept in memory. True will make the code faster but consumes more RAM memory.
        
        config = {"learning rate": lr,
                  "scattering learning rate": lr_sn,
                  "hidden": hidden,
                  "wd": wd,
                  "channels": channels,
                  "epochs": self.epochs,
                  "batch size": self.batch_size,
                  "network": self.arch,
                  "features": self.features,
                  "splits":self.splits}

        ## Initialise wandb
        wandb.login()
        wandb.init(project="%s" % self.name, entity="chris-pedersen",config=config)

        ## Set number of classes for scattering network to output

        # optimizer parameters
        beta1 = 0.5
        beta2 = 0.999
        #######################################################################################################
        #######################################################################################################

        # get training set
        print('\nPreparing training set')
        train_loader = create_dataset_multifield('train', seed, fmaps, fparams, batch_size, splits, fmaps_norm,
                                                 num_workers=self.num_workers, rot_flip_in_mem=rot_flip_in_mem, verbose=True)

        # get validation set
        print('\nPreparing validation set')
        valid_loader = create_dataset_multifield('valid', seed, fmaps, fparams, batch_size, splits, fmaps_norm,
                                                 num_workers=self.num_workers, rot_flip_in_mem=rot_flip_in_mem,  verbose=True)    

        # get test set
        print('\nPreparing test set')
        test_loader = create_dataset_multifield('test', seed, fmaps, fparams, batch_size, splits, fmaps_norm,
                                                num_workers=self.num_workers, rot_flip_in_mem=rot_flip_in_mem,  verbose=True)

        num_train_maps=len(train_loader.dataset.x)
        wandb.config.update({"no. training maps": num_train_maps,
                             "fields": fmaps})

        if self.arch=="sn":
            ## First create a scattering network object
            scatteringBase = baseModelFactory( #creat scattering base model
                architecture='scattering',
                J=2,
                N=256,
                M=256,
                channels=channels,
                max_order=2,
                initialization="Random",
                seed=234,
                learnable=True,
                lr_orientation=lr_sn,
                lr_scattering=lr_sn,
                skip=True,
                split_filters=True,
                filter_video=False,
                subsample=4,
                device=device,
                use_cuda=True
            )

            ## Now create a network to follow the scattering layers
            ## can be MLP, linear, or cnn at the moment
            ## (as in https://github.com/bentherien/ParametricScatteringNetworks/ )
            top = topModelFactory( #create cnn, mlp, linearlayer, or other
                base=scatteringBase,
                architecture="cnn",
                num_classes=features,
                width=hidden,
                average=False,
                use_cuda=True
            )

            ## Merge these into a hybrid model
            hybridModel = sn_HybridModel(scatteringBase=scatteringBase, top=top, use_cuda=True)
            model=hybridModel
            wandb.config.update({"learnable":scatteringBase.learnable,
                                 "initialisation":scatteringBase.initialization,
                                 "wavelet seed":scatteringBase.seed,
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
            print("setting up model %s" % self.arch)
            model = get_architecture(self.arch,hidden,dr,channels)
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
            if self.arch=="sn":
                wave_params=hybridModel.scatteringBase.params_filters
                orientations=wave_params[0].cpu().detach().numpy()
                xis=wave_params[1].cpu().detach().numpy()
                sigmas=wave_params[2].cpu().detach().numpy()
                slants=wave_params[3].cpu().detach().numpy()
                for aa in range(len(orientations)):
                    log_dic["orientation_%d" % aa]=orientations[aa]
                    log_dic["xi_%d" % aa]=xis[aa]
                    log_dic["sigma_%d" % aa]=sigmas[aa]
                    log_dic["slant_%d" % aa]=slants[aa]

            # train
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
                if self.features==4 or self.features==12:
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
            if self.features==4 or self.features==12:
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
                    if self.features==4 or self.features==12:    
                        e_NN  = p[:,h]         #posterior std
                        loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                        valid_loss2 += loss2*bs
                    valid_loss1 += loss1*bs
                    points     += bs


            valid_loss = torch.log(valid_loss1/points) 
            if self.features==4 or self.features==12:
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

    ## Model performance metrics on test set
    num_maps=test_loader.dataset.size
    ## Now loop over test set and print accuracy
    # define the arrays containing the value of the parameters
    params_true = np.zeros((num_maps,len(g)), dtype=np.float32)
    params_NN   = np.zeros((num_maps,len(g)), dtype=np.float32)
    errors_NN   = np.zeros((num_maps,len(g)), dtype=np.float32)
    
    # get test loss
    test_loss1, test_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    test_loss, points = 0.0, 0
    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            bs    = x.shape[0]         #batch size
            x     = x.to(device)       #send data to device
            y     = y.to(device)[:,g]  #send data to device
            p     = model(x)           #prediction for mean and variance
            y_NN  = p[:,g]             #prediction for mean
            loss1 = torch.mean((y_NN - y)**2,                axis=0)
            if self.features==4 or self.features==12:
                e_NN  = p[:,h]         #posterior std
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                test_loss2 += loss2*bs
            test_loss1 += loss1*bs
            test_loss = torch.log(test_loss1/points)
            if self.features==4 or self.features==12:
                test_loss+=torch.log(test_loss2/points)
            test_loss = torch.mean(test_loss).item()

	#e_NN  = p[:,6:]       #prediction for error
	#loss1 = torch.mean((y_NN[:,g] - y[:,g])**2,                     axis=0)
	#loss2 = torch.mean(((y_NN[:,g] - y[:,g])**2 - e_NN[:,g]**2)**2, axis=0)
	#test_loss1 += loss1*bs
	#test_loss2 += loss2*bs

            # save results to their corresponding arrays
            params_true[points:points+x.shape[0]] = y.cpu().numpy() 
            params_NN[points:points+x.shape[0]]   = y_NN.cpu().numpy()
            if self.features==4 or self.features==12:
                errors_NN[points:points+x.shape[0]]   = e_NN.cpu().numpy()
            points    += x.shape[0]
    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss = %.3e\n'%test_loss)
    
    # de-normalize
    ## I guess these are the hardcoded parameter limits
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    
    ## Drop feedback parameters if they aren't included
    minimum=minimum[g]
    maximum=maximum[g]
    params_true = params_true*(maximum - minimum) + minimum
    params_NN   = params_NN*(maximum - minimum) + minimum
    
    
    test_error = 100*np.mean(np.sqrt((params_true - params_NN)**2)/params_true,axis=0)
    print('Error Omega_m = %.3f'%test_error[0])
    print('Error sigma_8 = %.3f'%test_error[1])

    wandb.run.summary["Error Omega_m"]=test_error[0]
    wandb.run.summary["Error sigma_8"]=test_error[1]
    
    if self.features>4:
        print('Error A_SN1   = %.3f'%test_error[2])
        print('Error A_AGN1  = %.3f'%test_error[3])
        print('Error A_SN2   = %.3f'%test_error[4])
        print('Error A_AGN2  = %.3f\n'%test_error[5])
        wandb.run.summary["Error A_SN1"]  =test_error[2]
        wandb.run.summary["Error A_AGN1"] =test_error[3]
        wandb.run.summary["Error A_SN2"]  =test_error[4]
        wandb.run.summary["Error A_AGN2"] =test_error[5]
    
    wandb.run.summary["Error Omega_m"]=test_error[0]
    wandb.run.summary["Error sigma_8"]=test_error[1]
    
    if self.features==4:
        errors_NN   = errors_NN*(maximum - minimum)
        mean_error = 100*(np.absolute(np.mean(errors_NN/params_NN, axis=0)))
        print('Bayesian error Omega_m = %.3f'%mean_error[0])
        print('Bayesian error sigma_8 = %.3f'%mean_error[1])
        wandb.run.summary["Predicted error Omega_m"]=mean_error[0]
        wandb.run.summary["Predicted error sigma_8"]=mean_error[1]
    
    elif self.features==12:
        errors_NN   = errors_NN*(maximum - minimum)
        mean_error = 100*(np.absolute(np.mean(errors_NN/params_NN, axis=0)))
        print('Bayesian error Omega_m = %.3f'%mean_error[0])
        print('Bayesian error sigma_8 = %.3f'%mean_error[1])
        print('Bayesian error A_SN1   = %.3f'%mean_error[2])
        print('Bayesian error A_AGN1  = %.3f'%mean_error[3])
        print('Bayesian error A_SN2   = %.3f'%mean_error[4])
        print('Bayesian error A_AGN2  = %.3f\n'%mean_error[5])
        wandb.run.summary["Predicted error Omega_m"]=mean_error[0]
        wandb.run.summary["Predicted error sigma_8"]=mean_error[1]
        wandb.run.summary["Predicted error A_SN1"]  =mean_error[2]
        wandb.run.summary["Predicted error A_AGN1"] =mean_error[3]
        wandb.run.summary["Predicted error A_SN2"]  =mean_error[4]
        wandb.run.summary["Predicted error A_AGN2"] =mean_error[5]
    
    
    if self.features<5:
        f, axarr = plt.subplots(1, 2, figsize=(9,6))
        axarr[0].plot(np.linspace(min(params_true[:,0]),max(params_true[:,0]),100),np.linspace(min(params_true[:,0]),max(params_true[:,0]),100),color="black")
        axarr[1].plot(np.linspace(min(params_true[:,1]),max(params_true[:,1]),100),np.linspace(min(params_true[:,1]),max(params_true[:,1]),100),color="black")
        if self.features==4:
            axarr[0].errorbar(params_true[:,0],params_NN[:,0],errors_NN[:,0],marker="o",ls="none")
            axarr[1].errorbar(params_true[:,1],params_NN[:,1],errors_NN[:,1],marker="o",ls="none")
        else:
            axarr[0].plot(params_true[:,0],params_NN[:,0],marker="o",ls="none")
            axarr[1].plot(params_true[:,1],params_NN[:,1],marker="o",ls="none")
            
        axarr[0].set_xlabel(r"True $\Omega_m$")
        axarr[0].set_ylabel(r"Predicted $\Omega_m$")
        axarr[0].text(0.1,0.9,"%.3f %% error" % test_error[0],fontsize=12,transform=axarr[0].transAxes)

        axarr[1].set_xlabel(r"True $\sigma_8$")
        axarr[1].set_ylabel(r"Predicted $\sigma_8$")
        axarr[1].text(0.1,0.9,"%.3f %% error" % test_error[1],fontsize=12,transform=axarr[1].transAxes)


    if self.features>4:
        f, axarr = plt.subplots(3, 2, figsize=(14,20))
        for aa in range(0,6,2):
            axarr[aa//2][0].plot(np.linspace(min(params_true[:,aa]),max(params_true[:,aa]),100),np.linspace(min(params_true[:,aa]),max(params_true[:,aa]),100),color="black")
            axarr[aa//2][1].plot(np.linspace(min(params_true[:,aa+1]),max(params_true[:,aa+1]),100),np.linspace(min(params_true[:,aa+1]),max(params_true[:,aa+1]),100),color="black")
            if self.features==12:
                axarr[aa//2][0].errorbar(params_true[:,aa],params_NN[:,aa],errors_NN[:,aa],marker="o",ls="none")
                axarr[aa//2][1].errorbar(params_true[:,aa+1],params_NN[:,aa+1],errors_NN[:,aa+1],marker="o",ls="none")
            else:
                axarr[aa//2][0].plot(params_true[:,aa],params_NN[:,aa],marker="o",ls="none")
                axarr[aa//2][1].plot(params_true[:,aa+1],params_NN[:,aa+1],marker="o",ls="none")
                 
        axarr[0][0].set_xlabel(r"True $\Omega_m$")
        axarr[0][0].set_ylabel(r"Predicted $\Omega_m$")
        axarr[0][0].text(0.1,0.9,"%.3f %% error" % test_error[0],fontsize=12,transform=axarr[0][0].transAxes)

        axarr[0][1].set_xlabel(r"True $\sigma_8$")
        axarr[0][1].set_ylabel(r"Predicted $\sigma_8$")
        axarr[0][1].text(0.1,0.9,"%.3f %% error" % test_error[1],fontsize=12,transform=axarr[0][1].transAxes)

        axarr[1][0].set_xlabel(r"True $A_\mathrm{SN1}$")
        axarr[1][0].set_ylabel(r"Predicted $A_\mathrm{SN1}$")
        axarr[1][0].text(0.1,0.9,"%.3f %% error" % test_error[2],fontsize=12,transform=axarr[1][0].transAxes)

        axarr[1][1].set_xlabel(r"True $A_\mathrm{AGN1}$")
        axarr[1][1].set_ylabel(r"Predicted $A_\mathrm{AGN1}$")
        axarr[1][1].text(0.1,0.9,"%.3f %% error" % test_error[3],fontsize=12,transform=axarr[1][1].transAxes)

        axarr[2][0].set_xlabel(r"True $A_\mathrm{SN2}$")
        axarr[2][0].set_ylabel(r"Predicted $A_\mathrm{SN2}$")
        axarr[2][0].text(0.1,0.9,"%.3f %% error" % test_error[4],fontsize=12,transform=axarr[2][0].transAxes)

        axarr[2][1].set_xlabel(r"True $A_\mathrm{AGN2}$")
        axarr[2][1].set_ylabel(r"Predicted $A_\mathrm{AGN2}$")
        axarr[2][1].text(0.1,0.9,"%.3f %% error" % test_error[4],fontsize=12,transform=axarr[2][1].transAxes)

    figure=wandb.Image(f)
    wandb.log({"performance": figure})
    wandb.finish()

        return valid_loss

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

## camels path

# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_B.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_HI.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Mcdm.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Mgas.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_MgFe.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Mstar.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Mtot.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_ne.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_P.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_T.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Vcdm.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Vgas.npy"
# "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Z.npy"

camels_path="/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/2D_maps/data/"
fparams    = camels_path+"/params_IllustrisTNG.txt"
fmaps      = ["/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_B.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_HI.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Mgas.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_MgFe.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Mstar.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Mtot.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_ne.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_P.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_T.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Vgas.npy",
              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/1k_fields/maps_Z.npy"              
             ]
#fmaps      = [
#              "/mnt/home/cpedersen/ceph/Data/CAMELS_test/15k_fields/maps_Mtot.npy"         
#             ]
fmaps_norm      = [None]
splits          = 1
seed            = 123
params          = [0,1,2,3,4,5] #0(Om) 1(s8) 2(A_SN1) 3 (A_AGN1) 4(A_SN2) 5(A_AGN2)
monopole        = True  #keep the monopole of the maps (True) or remove it (False)
rot_flip_in_mem = False  #whether rotations and flipings are kept in memory
smoothing       = 0  ## Smooth the maps with a Gaussian filter? 0 for no
arch            = "a" ## Which model architecture to use
features        = 2
name            = "test_optuna_lossdrop" ## For wandb and optuna

## training parameters
batch_size  = 32
epochs      = 15
num_workers = 10    #number of workers to load data

## Optuna params
study_name = "optuna/%s" % name  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
n_trials=20

# train networks with bayesian optimization
objective = Objective(device, seed, fmaps, fmaps_norm, fparams, batch_size, splits,
                      arch, error, beta1, beta2, epochs, monopole, 
                      num_workers, params, rot_flip_in_mem, smoothing, name)
sampler = optuna.samplers.TPESampler(n_startup_trials=20)
study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage_name,
                            load_if_exists=True)
study.optimize(objective, n_trials, gc_after_trial=False)

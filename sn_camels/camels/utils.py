import numpy as np
import sys, os, time, optuna


# This routine reads a database and returns the value of the best-trial
def best_params_database(study_name, storage):
    
    # load the optuna study
    study = optuna.load_study(study_name=study_name, storage=storage)

    # get the scores of the study trials
    values = np.zeros(len(study.trials))
    completed = 0
    for i,t in enumerate(study.trials):
        values[i] = t.value
        if t.value is not None:  completed += 1

    # get the info of the best trial
    indexes = np.argsort(values)
    for i in [0]: #range(1):
        trial = study.trials[indexes[i]]
        print("\nTrial number {}".format(trial.number))
        print("Value: %.5e"%trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        num    = trial.number 
        dr     = trial.params['dr']
        hidden = trial.params['hidden']
        lr     = trial.params['lr']
        wd     = trial.params['wd']

    return num, dr, hidden, lr, wd


# This routine returns the name of the file with the best-model and its parameters
def best_model_params(study_name, storage, fields, root, label, errors=False, 
                      sim='IllustrisTNG', monopole=True):

    # get the value of the parameters of the best-model
    num, dr, hidden, lr, wd = best_params_database(study_name, storage)

    # determine whether it is a multifield or not
    multifield = False
    if len(fields)>1:  multifield = True

    # get the suffix of the file name
    suffix = ''
    if multifield:
        for field in fields:
            suffix = '%s_%s'%(suffix,field)
    else:
        suffix = '_%s'%fields[0]
    if not(monopole):  suffix = '%s_no_monopole'%suffix

    # get the name of the files
    if errors is False:
        if label is None:
            fmodel = '%s/models/model%s_cosmo_%s.pt'%(root, suffix, num)
        else:
            fmodel = '%s/models/model%s_cosmo_%s_%s.pt'%(root, suffix, num, label)
    elif errors is True:
        if label is None:
            fmodel = '%s/models/model%s_%s_errors.pt'%(root, suffix, num)
        else:
            fmodel = '%s/models/model%s_%s_%s_errors.pt'%(root, suffix, num, label)
    elif errors=='new':
        if label is None:
            fmodel = '%s/models_%s/model%s_%s.pt'%(root, sim, suffix, num)
        else:
            fmodel = '%s/models_%s/model%s_%s_%s.pt'%(root, sim, suffix, num, label)

    return fmodel, num, dr, hidden, lr, wd


# This routine returns the number of maps in a data loader
def dataloader_elements(loader):

    # get the number of maps in the test set
    num_maps = 0
    for x,y in loader:
        num_maps += x.shape[0]

    return num_maps


# This routine returns the file name containing the maps of a given field and its suffix
# root ---------> folder containing the maps
# field --------> tuple with the considered fields
# sim ----------> 'IllustrisTNG' or 'SIMBA'
# z ------------> redshift
# norm ---------> whether normalize the fields or not
# suite --------> 'LH' or 'CV'
# sim_norm------> if training on TNG and predicting on SIMBA set sim_norm='IllustrisTNG'
def fname_maps(root, fields, sim, z, suite='LH', sim_norm=None):

    # do a loop over the different fields
    f_maps, f_maps_norm, suffix = [], [], ''
    for field in fields:
        suffix = '%s_%s'%(suffix,field)
        f_maps.append('%s/Images_%s_%s_%s_z=%.2f.npy'%(root,field,sim,suite,z))
        if sim_norm is None:
            if suite=='LH':
                f_maps_norm.append(None)
            else:
                f_maps_norm.append('%s/Images_%s_%s_LH_z=%.2f.npy'%(root,field,sim,z))
        else:
            f_maps_norm.append('%s/Images_%s_%s_LH_z=%.2f.npy'%(root,field,sim_norm,z))

    return f_maps, f_maps_norm, suffix


# This routine returns the name of the storage databases
def fname_storage(root, fields, label_m, label_e=None, monopole=True):
    
    # do a loop over the different fields
    storage = '%s'%root
    for field in fields:
        storage = '%s_%s'%(storage,field)
    if monopole is False:  storage = '%s_no_monopole'%storage

    # add the labels
    if label_m is None:  
        storage_m = '%s.db'%storage
        storage_e = '%s_errors.db'%storage
    else:  
        storage_m = '%s_%s.db'%(storage, label_m)
        storage_e = '%s_%s_errors.db'%(storage, label_e)

    if label_e is None:  return storage_m
    else:                return storage_m, storage_e


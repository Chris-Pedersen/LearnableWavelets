import numpy as np
import sys, os, time
import torch 
import torch.nn as nn
import data
import architecture
import optuna
import utils as U

def get_results(study_name, storage, arch, channels, device, root_files, test_loader,
                suite, suffix_train, label_train, subfields_train, monopole_train,
                suffix_test, label_test, subfields_test, monopole_test):

    ####### load best model ######## Training
    # get the parameters of the best model for the mean
    fmodel, num, dr, hidden, lr, wd = U.best_model_params(study_name, storage, 
                                                          subfields_train, root_files, 
                                                          label_train, 'new', 
                                                          sim_train, monopole_train)

    # get the model
    model = architecture.get_architecture(arch+'_err', hidden, dr, channels)
    model = nn.DataParallel(model)
    model.to(device=device)
    network_total_params = sum(p.numel() for p in model.parameters())
    print('total number of parameters in the model = %d'%network_total_params)

    # load best-model, if it exists
    if os.path.exists(fmodel):  
        print('Loading model...')
        model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
    else:
        raise Exception('model doesnt exists!!!')
    ################################

    # get the name of the output files
    suffix1 = 'train_%s%s_%s'%(sim_train, suffix_train, label_train)
    if not(monopole_train):  suffix1 = '%s_no_monopole'%suffix1
    suffix2 = 'test_%s%s_%s'%(sim_test, suffix_test, label_test)
    if not(monopole_test):   suffix2 = '%s_no_monopole'%suffix2
    suffix = '%s_%s_%s.txt'%(suffix1, suffix2, suite)
    fresults  = 'results/results_%s'%suffix   
    fresults1 = 'results/Normalized_errors_%s'%suffix

    # get the number of maps in the test set
    num_maps = U.dataloader_elements(test_loader)
    print('\nNumber of maps in the test set: %d'%num_maps)

    # define the arrays containing the value of the parameters
    params_true = np.zeros((num_maps,6), dtype=np.float32)
    params_NN   = np.zeros((num_maps,6), dtype=np.float32)
    errors_NN   = np.zeros((num_maps,6), dtype=np.float32)

    # get test loss
    g = [0, 1, 2, 3, 4, 5]
    test_loss1, test_loss2 = torch.zeros(len(g)).to(device), torch.zeros(len(g)).to(device)
    test_loss, points = 0.0, 0
    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
            bs    = x.shape[0]    #batch size
            x     = x.to(device)  #send data to device
            y     = y.to(device)  #send data to device
            p     = model(x)      #prediction for mean and variance
            y_NN  = p[:,:6]       #prediction for mean
            e_NN  = p[:,6:]       #prediction for error
            loss1 = torch.mean((y_NN[:,g] - y[:,g])**2,                     axis=0)
            loss2 = torch.mean(((y_NN[:,g] - y[:,g])**2 - e_NN[:,g]**2)**2, axis=0)
            test_loss1 += loss1*bs
            test_loss2 += loss2*bs

            # save results to their corresponding arrays
            params_true[points:points+x.shape[0]] = y.cpu().numpy() 
            params_NN[points:points+x.shape[0]]   = y_NN.cpu().numpy()
            errors_NN[points:points+x.shape[0]]   = e_NN.cpu().numpy()
            points    += x.shape[0]
    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss = %.3e\n'%test_loss)

    Norm_error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    print('Normalized Error Omega_m = %.3f'%Norm_error[0])
    print('Normalized Error sigma_8 = %.3f'%Norm_error[1])
    print('Normalized Error A_SN1   = %.3f'%Norm_error[2])
    print('Normalized Error A_AGN1  = %.3f'%Norm_error[3])
    print('Normalized Error A_SN2   = %.3f'%Norm_error[4])
    print('Normalized Error A_AGN2  = %.3f\n'%Norm_error[5])

    # de-normalize
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
    params_true = params_true*(maximum - minimum) + minimum
    params_NN   = params_NN*(maximum - minimum) + minimum
    errors_NN   = errors_NN*(maximum - minimum)

    error = np.sqrt(np.mean((params_true - params_NN)**2, axis=0))
    print('Error Omega_m = %.3f'%error[0])
    print('Error sigma_8 = %.3f'%error[1])
    print('Error A_SN1   = %.3f'%error[2])
    print('Error A_AGN1  = %.3f'%error[3])
    print('Error A_SN2   = %.3f'%error[4])
    print('Error A_AGN2  = %.3f\n'%error[5])

    mean_error = np.absolute(np.mean(errors_NN, axis=0))
    print('Bayesian error Omega_m = %.3f'%mean_error[0])
    print('Bayesian error sigma_8 = %.3f'%mean_error[1])
    print('Bayesian error A_SN1   = %.3f'%mean_error[2])
    print('Bayesian error A_AGN1  = %.3f'%mean_error[3])
    print('Bayesian error A_SN2   = %.3f'%mean_error[4])
    print('Bayesian error A_AGN2  = %.3f\n'%mean_error[5])

    rel_error = np.sqrt(np.mean((params_true - params_NN)**2/params_true**2, axis=0))
    print('Relative error Omega_m = %.3f'%rel_error[0])
    print('Relative error sigma_8 = %.3f'%rel_error[1])
    print('Relative error A_SN1   = %.3f'%rel_error[2])
    print('Relative error A_AGN1  = %.3f'%rel_error[3])
    print('Relative error A_SN2   = %.3f'%rel_error[4])
    print('Relative error A_AGN2  = %.3f\n'%rel_error[5])

    # save results to file
    dataset = np.zeros((num_maps,18), dtype=np.float32)
    dataset[:,:6]   = params_true
    dataset[:,6:12] = params_NN
    dataset[:,12:]  = errors_NN
    np.savetxt(fresults,  dataset)
    np.savetxt(fresults1, Norm_error)




##################################### INPUT ##########################################
# architecture parameters
arch = 'o3'

# properties of the network/maps used for training
sim_train       = 'IllustrisTNG'
fields_train    = ['T']
#['Mgas','Mcdm','Mtot','Mstar','Vgas','Vcdm',
                  # 'HI', 'ne', 'P', 'T']  
monopole_train  = True
smoothing_train = 0
label_train     = 'all_steps_500_500_%s'%arch 

# properties of the maps for testing
sim_test       = 'SIMBA'  
fields_test    = ['T']
#['Mgas','Mcdm','Mtot','Mstar','Vgas','Vcdm',
#                  'HI', 'ne', 'P', 'T']  
monopole_test  = True
smoothing_test = 0
label_test     = 'all_steps_500_500_%s'%arch     
suite          = 'LH' #'LH' or 'CV'
mode           = 'test'

# other parameters (usually no need to modify)
root_files    = '/mnt/ceph/users/camels/Results/multifield_2_params'
root_maps     = '/mnt/ceph/users/camels/Results/images_new' #folder containing the maps
batch_size    = 32
root_storage  = 'sqlite:///databases_%s/%s'%(sim_train,arch)
study_name    = 'wd_dr_hidden_lr_%s'%arch  
z             = 0.00  #redshift
seed          = 1               #random seed to initially mix the maps
splits        = 15              #number of maps per simulation
just_monopole = False #whether to create the images with just the monopole
sim_norm      = sim_train

# data parameters
#['Comb', 'Mgas', 'Mcdm', 'Mtot', 'Mstar', 'Vgas', 'Vcdm',
#                'T', 'Z', 'HI', 'P', 'ne', 'B', 'MgFe',
#                'Nbody', 'Comb']
######################################################################################

if smoothing_train>0:  label_train = '%s_smoothing_%d'%(label_train,smoothing_train)
if smoothing_test>0:   label_test  = '%s_smoothing_%d'%(label_test, smoothing_train)

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# define loss function
criterion = nn.MSELoss() 

# get the test mode and the file with the parameters
if suite=='LH':    
    f_params = '/mnt/ceph/users/camels/Software/%s/latin_hypercube_params.txt'%sim_test
elif suite=='CV': 
    f_params = '/mnt/ceph/users/camels/Software/%s/CV_params.txt'%sim_test
else:          raise Exception('wrong suite value')

# do a loop over the different fields
for field_train, field_test in zip(fields_train, fields_test):

    print('\n############# Trained on %s %s ##################'%(field_train,sim_train))
    print('############# Testing on %s %s ##################'%(field_test,sim_test))

    ############ Training files #############
    # get the subfields of the considered field
    if field_train=='Comb':  subfields_train = ['Mgas', 'Mtot', 'Mstar', 'Vgas', 'T',
                                                'Z', 'P', 'HI', 'ne', 'B', 'MgFe']
    else:                    subfields_train = [field_train]

    # get the suffix
    suffix_train = ''
    for x in subfields_train:  suffix_train = '%s_%s'%(suffix_train,x)

    # get the storage name to use best-model (training)
    storage = U.fname_storage(root_storage, subfields_train, label_train, 
                              monopole=monopole_train)
    #########################################

    ########## Testing files ################
    # get the subfields of the considered field
    if field_test=='Comb':   subfields_test = ['Mgas', 'Mtot', 'Mstar', 'Vgas', 'T', 
                                               'Z', 'P', 'HI', 'ne', 'B', 'MgFe']
    else:                    subfields_test = [field_test]

    # get the name of the maps to test the model
    channels = len(subfields_test)
    f_maps, f_maps_norm, suffix_test = U.fname_maps(root_maps, subfields_test, sim_test,
                                                    z, suite, sim_norm)

    # get the test data
    test_loader = data.create_dataset_multifield(mode, seed, f_maps, f_params, 
                            batch_size, splits, f_maps_norm, monopole=monopole_test, 
                            monopole_norm=monopole_train, rot_flip_in_mem=True, 
                            shuffle=False, just_monopole=just_monopole, 
                            smoothing=smoothing_test, smoothing_norm=smoothing_train, 
                            verbose=True)
    #########################################

    # get the results
    
    get_results(study_name, storage, arch, channels, device, root_files, test_loader,
                suite, suffix_train, label_train, subfields_train, monopole_train,
                suffix_test, label_test, subfields_test, monopole_test)
                
    print('############################################')












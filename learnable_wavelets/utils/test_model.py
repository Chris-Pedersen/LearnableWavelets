import torch
import numpy as np

def test_model(model,test_loader,device):
    """ Simple script to take a model and a camels dataset
    and loop over all samples to test the accuracy of the model.
    NB we do not ensure that the test set is different to the training set
    inside this script. """

    num_maps=test_loader.dataset.size
    ## Now loop over test set and print accuracy
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
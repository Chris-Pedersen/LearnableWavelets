"""Contains the factories for selecting different models

Author: Benjamin Therien

Functions: 
    baseModelFactory -- Factory for the creation of the first part of a hybrid model
    topModelFactory -- Factory for the creation of seconds part of a hybrid model

Exceptions:
    InvalidArchitectureError -- Error thrown when an invalid architecture name is passed
"""


from sn_camels.models.sn_base_models import sn_Identity, sn_ScatteringBase
from sn_camels.models.sn_top_models import sn_CNN, sn_MLP, sn_LinearLayer, sn_Resnet50

class InvalidArchitectureError(Exception):
    """Error thrown when an invalid architecture name is passed"""
    pass


def baseModelFactory(architecture, J, N, M, max_order, initialization, seed, device, 
                     learnable=True, lr_orientation=0.1, lr_scattering=0.1, skip=True,
                     split_filters=False, subsample=1, filter_video=False,
                     use_cuda=True):
    """Factory for the creation of the first layer of a hybrid model
    
        parameters: 
            J -- scale of scattering (always 2 for now)
            N -- height of the input image
            M -- width of the input image
            second_order -- 
            initilization -- the type of init: ['Tight-Frame' or 'Random']
            seed -- the random seed used to initialize the parameters
            device -- the device to place weights on
            learnable -- should the filters be learnable parameters of this model
            lr_orientation -- learning rate for the orientation of the scattering parameters
            lr_scattering -- learning rate for scattering parameters other than orientation
            skip -- whether or not to include skip connections when using learnable filters
            split_filters -- split first and second order filters      
            subsample -- amount to subsample the output fields           
            monitor_filters -- boolean indicating whether to track filter distances from initialization
            use_cuda -- True if using GPU
    """

    if architecture.lower() == 'identity':
        return sn_Identity()

    elif architecture.lower() == 'scattering':
        return sn_ScatteringBase( #create learnable of non-learnable scattering
            J=J,
            N=N,
            M=M,
            max_order=max_order,
            initialization=initialization,
            seed=seed,
            learnable=learnable,
            lr_orientation=lr_orientation,
            lr_scattering=lr_scattering,
            skip=skip,
            split_filters=split_filters,
            subsample=subsample,
            filter_video=filter_video,
            device=device,
            use_cuda=use_cuda
        )

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()



def topModelFactory(base, architecture, num_classes, width=8, average=False, use_cuda=True):
    """Factory for the creation of seconds part of a hybrid model
    
    parameters:
        base         -- (Pytorch nn.Module) the first part of a hybrid model
        architecture -- the name of the top model to select
        num_classes  -- number of classes in dataset
        width        -- the width of the model
        average      -- boolean indicating whether to average the spatial information 
                        of the scattering coefficients
        use_cuda     -- boolean indicating whether to use GPU
    """

    if architecture.lower() == 'cnn':
        return sn_CNN(
            base.n_coefficients, k=width, num_classes=num_classes, standard=False
        )

    elif architecture.lower() == 'mlp':
        return sn_MLP(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
            use_cuda=use_cuda
        )

    elif architecture.lower() == 'linear_layer':
        return sn_LinearLayer(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
        )

    elif architecture.lower() == 'resnet50':
        return sn_Resnet50(num_classes=num_classes)

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()
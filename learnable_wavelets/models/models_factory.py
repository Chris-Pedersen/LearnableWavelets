from learnable_wavelets.models.sn_base_models import sn_Identity, sn_ScatteringBase
from learnable_wavelets.models.sn_top_models import sn_CNN, sn_MLP, sn_LinearLayer, sn_Resnet50

class InvalidArchitectureError(Exception):
    """Error thrown when an invalid architecture name is passed"""
    pass


def baseModelFactory(architecture, J, N, M, channels, max_order, initialization, seed, device, 
                     learnable=True, lr_orientation=0.1, lr_scattering=0.1, skip=True,
                     split_filters=False, subsample=1,
                     use_cuda=True,plot=True):
    """ Factory for the creation of the first layer of a hybrid model
            J              -- Ccale of scattering (always 2 for now - this parameter is being phased out)
            N              -- Height of the input image
            M              -- Width of the input image
            channels       -- Number of different input fields
            max_order      -- Highest order of wavelet scattering
            initilization  -- Wavelet initialisation ['Tight-Frame' or 'Random']
            seed           -- The random seed used to initialize the parameters
            device         -- The device to place weights on
            learnable      -- Learn wavelet parameters, bool
            lr_orientation -- Learning rate for the orientation of the scattering parameters
            lr_scattering  -- Learning rate for scattering parameters other than orientation
            skip           -- Whether or not to include skip connections when using learnable filters
            split_filters  -- True to use different wavelets for first and second order scattering     
            subsample      -- Amount of downsampling at each wavelet convolution step
            use_cuda       -- True if using GPU
            plot           -- Plot wavelets when creating the scattering module
    """

    if architecture.lower() == 'identity':
        return sn_Identity()

    elif architecture.lower() == 'scattering':
        return sn_ScatteringBase( #create learnable of non-learnable scattering
            J=J,
            N=N,
            M=M,
            channels=channels,
            max_order=max_order,
            initialization=initialization,
            seed=seed,
            learnable=learnable,
            lr_orientation=lr_orientation,
            lr_scattering=lr_scattering,
            skip=skip,
            split_filters=split_filters,
            subsample=subsample,
            device=device,
            use_cuda=use_cuda,
            plot=plot
        )

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()


def topModelFactory(base, architecture, num_classes, width=8, average=False, use_cuda=True):
    """ Factory for the creation of second part of a hybrid model
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
            base.n_coefficients*base.channels, k=width, num_classes=num_classes
        )

    elif architecture.lower() == 'mlp':
        return sn_MLP(
            num_classes=num_classes, n_coefficients=base.n_coefficients*base.channels, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
            use_cuda=use_cuda
        )

    elif architecture.lower() == 'linear_layer':
        return sn_LinearLayer(
            num_classes=num_classes, n_coefficients=base.n_coefficients*base.channels, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient,
            average=average, use_cuda=use_cuda
        )

    elif architecture.lower() == 'resnet50':
        return sn_Resnet50(num_classes=num_classes)

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()
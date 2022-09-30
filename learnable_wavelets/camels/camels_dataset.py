import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time
import torchvision.utils as vutils
import scipy.ndimage
import learnable_wavelets.camels.utils as U


# This routine takes set of maps and smooth them with a Gaussian kernel
def smooth_maps(maps, smoothing, verbose=True):
    
    if verbose:  print('Smoothing images with smoothing length: %d'%smoothing)

    # do a loop over all maps
    for i in range(maps.shape[0]):
        image = maps[i]
        maps[i] = scipy.ndimage.gaussian_filter(image, smoothing, mode='wrap')

    return maps

# This routine takes a set of maps and remove their monopole (i.e. average value)
def remove_monopole(maps, verbose=True):

    if verbose:  print('removing monopoles')

    # compute the mean of each map
    maps_mean = np.mean(maps, axis=(1,2), dtype=np.float64)

    # do a loop over all maps and remove mean value
    for i in range(maps.shape[0]):
        maps[i] = maps[i] - maps_mean[i]

    return maps

# This class creates the dataset. It will read the maps and store them in memory
# the rotations and flipings are done when calling the data 
class make_dataset_multifield2():

    def __init__(self, mode, seed, f_images, f_params, splits, f_images_norm, 
                 monopole, monopole_norm, smoothing, smoothing_norm, verbose):

        # get the total number of sims and maps
        params_sims = np.loadtxt(f_params) #simulations parameters, NOT maps parameters
        total_sims, total_maps, num_params = \
                params_sims.shape[0], params_sims.shape[0]*splits, params_sims.shape[1]
        params = np.zeros((total_maps, num_params), dtype=np.float32)
        for i in range(total_sims):
            for j in range(splits):
                params[i*splits + j] = params_sims[i]

        # normalize params
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params  = (params - minimum)/(maximum - minimum)

        # get the size and offset depending on the type of dataset
        if   mode=='train':  
            offset, size_sims = int(0.00*total_sims), int(0.90*total_sims)
        elif mode=='valid':  
            offset, size_sims = int(0.90*total_sims), int(0.05*total_sims)
        elif mode=='test':  
            offset, size_sims = int(0.95*total_sims), int(0.05*total_sims)
        elif mode=='all':  
            offset, size_sims = int(0.00*total_sims), int(1.00*total_sims)
        else:    raise Exception('Wrong name!')
        size_maps = size_sims*splits

        # randomly shuffle the simulations (not maps). Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        sim_numbers = np.arange(total_sims) #shuffle maps not rotations
        np.random.shuffle(sim_numbers)
        sim_numbers = sim_numbers[offset:offset+size_sims] #select indexes of mode

        # get the corresponding indexes of the maps associated to the sims
        indexes = np.zeros(size_maps, dtype=np.int32)
        count = 0
        for i in sim_numbers:
            for j in range(splits):
                indexes[count] = i*splits + j
                count += 1

        # keep only the value of the parameters of the considered maps
        params = params[indexes]

        # define the matrix containing the maps without rotations or flippings
        channels = len(f_images)
        dumb     = np.load(f_images[0])    #[number of maps, height, width]
        height, width = dumb.shape[1], dumb.shape[2];  del dumb
        data     = np.zeros((size_maps, channels, height, width), dtype=np.float32)

        # read the data
        print('Found %d channels\nReading data...'%channels)
        for channel, (fim, fnorm) in enumerate(zip(f_images, f_images_norm)):

            # read maps in the considered channel
            data_c = np.load(fim)
            if data_c.shape[0]!=total_maps:  raise Exception('sizes do not match')
            if verbose:  
                print('%.3e < F(all|orig) < %.3e'%(np.min(data_c), np.max(data_c)))

            # smooth the images
            if smoothing>0:  data_c = smooth_maps(data_c, smoothing, verbose)

            # rescale maps
            if fim.find('Mstar')!=-1:  data_c = np.log10(data_c + 1.0)
            else:                      data_c = np.log10(data_c)
            if verbose:  
                print('%.3f < F(all|resc)  < %.3f'%(np.min(data_c), np.max(data_c)))

            # remove monopole of the images
            if monopole is False:  data_c = remove_monopole(data_c, verbose)

            # normalize maps
            if fnorm is None:  
                mean,    std     = np.mean(data_c), np.std(data_c)
                minimum, maximum = np.min(data_c),  np.max(data_c)
            else:
                # read data
                data_norm     = np.load(fnorm)

                # smooth data
                if smoothing_norm>0:  
                    data_norm = smooth_maps(data_norm, smoothing_norm, verbose)

                # rescale data
                if fnorm.find('Mstar')!=-1:  data_norm = np.log10(data_norm + 1.0)
                else:                        data_norm = np.log10(data_norm)

                # remove monopole
                if monopole_norm is False:
                    data_norm = remove_monopole(data_norm, verbose)

                # compute mean and std
                mean,    std     = np.mean(data_norm), np.std(data_norm)
                minimum, maximum = np.min(data_norm),  np.max(data_norm)
                del data_norm

            #data = 2*(data - minimum)/(maximum - minimum) - 1.0
            data_c = (data_c - mean)/std
            if verbose:  
                print('%.3f < F(all|norm) < %.3f'%(np.min(data_c), np.max(data_c))) 

            # keep only the data of the chosen set
            data[:,channel,:,:] = data_c[indexes]

            #if verbose:
            #    print('Channel %d contains %d maps'%(channel,counted_maps))
            #    print('%.3f < F < %.3f\n'%(np.min(data_c), np.max(data_c)))
        
        self.size = data.shape[0]
        self.x    = torch.tensor(data,   dtype=torch.float32)
        self.y    = torch.tensor(params, dtype=torch.float32)
        del data, data_c

        #vutils.save_image(self.x, 'images.png', nrow=10, normalize=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        # choose a rotation angle (0-0, 1-90, 2-180, 3-270)
        # and whether do flipping or not
        rot  = np.random.randint(0,4)
        flip = np.random.randint(0,1)
        
        # rotate and flip the maps
        maps = torch.rot90(self.x[idx], k=rot, dims=[1,2])
        if flip==1:  maps = torch.flip(maps, dims=[1])

        return maps, self.y[idx]


# This class creates the dataset. Rotations and flippings are done and stored
class make_dataset_multifield():

    def __init__(self, mode, seed, f_images, f_params, splits, f_images_norm, 
                 monopole, monopole_norm, just_monopole, smoothing, smoothing_norm, 
                 verbose):

        # get the total number of sims and maps
        params_sims = np.loadtxt(f_params) #simulations parameters, NOT maps parameters
        total_sims, total_maps, num_params = \
                params_sims.shape[0], params_sims.shape[0]*splits, params_sims.shape[1]
        params_maps = np.zeros((total_maps, num_params), dtype=np.float32)
        for i in range(total_sims):
            for j in range(splits):
                params_maps[i*splits + j] = params_sims[i]

        # normalize params
        minimum     = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum     = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params_maps = (params_maps - minimum)/(maximum - minimum)

        # get the size and offset depending on the type of dataset
        if   mode=='train':  
            offset, size_sims = int(0.00*total_sims), int(0.90*total_sims)
        elif mode=='valid':  
            offset, size_sims = int(0.90*total_sims), int(0.05*total_sims)
        elif mode=='test':  
            offset, size_sims = int(0.95*total_sims), int(0.05*total_sims)
        elif mode=='all':  
            offset, size_sims = int(0.00*total_sims), int(1.00*total_sims)
        else:    raise Exception('Wrong name!')
        size_maps = size_sims*splits

        # randomly shuffle the simulations (not maps). Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        sim_numbers = np.arange(total_sims) #shuffle sims not maps
        np.random.shuffle(sim_numbers)
        sim_numbers = sim_numbers[offset:offset+size_sims] #select indexes of mode

        # get the corresponding indexes of the maps associated to the sims
        indexes = np.zeros(size_maps, dtype=np.int32)
        count = 0
        for i in sim_numbers:
            for j in range(splits):
                indexes[count] = i*splits + j
                count += 1

        # keep only the value of the parameters of the considered maps
        params_maps = params_maps[indexes]

        # define the matrix containing the maps with rotations and flipings
        channels = len(f_images)
        dumb     = np.load(f_images[0])    #[number of maps, height, width]
        height, width = dumb.shape[1], dumb.shape[2];  del dumb
        data     = np.zeros((size_maps*8, channels, height, width), dtype=np.float32)
        params   = np.zeros((size_maps*8, num_params),              dtype=np.float32)

        # read the data
        print('Found %d channels\nReading data...'%channels)
        for channel, (fim, fnorm) in enumerate(zip(f_images, f_images_norm)):

            # read maps in the considered channel
            data_c = np.load(fim)
            if data_c.shape[0]!=total_maps:  raise Exception('sizes do not match')
            if verbose:  
                print('%.3e < F(all|orig) < %.3e'%(np.min(data_c), np.max(data_c)))

            # smooth the images
            if smoothing>0:  data_c = smooth_maps(data_c, smoothing)

            # rescale maps
            if fim.find('Mstar')!=-1:  data_c = np.log10(data_c + 1.0)
            else:                      data_c = np.log10(data_c)
            if verbose:  
                print('%.3f < F(all|resc)  < %.3f'%(np.min(data_c), np.max(data_c)))

            # remove monopole of the images
            if monopole is False:  data_c = remove_monopole(data_c, verbose)

            # normalize maps
            if fnorm is None:  
                mean,    std     = np.mean(data_c), np.std(data_c)
                minimum, maximum = np.min(data_c),  np.max(data_c)
            else:
                # read data
                data_norm = np.load(fnorm)

                # smooth maps
                if smoothing_norm>0:  
                    data_norm = smooth_maps(data_norm, smoothing_norm, verbose)

                # rescale
                if fnorm.find('Mstar')!=-1:  data_norm = np.log10(data_norm + 1.0)
                else:                        data_norm = np.log10(data_norm)

                # remove monopole
                if monopole_norm is False:
                    data_norm = remove_monopole(data_norm, verbose)

                # compute mean and std
                mean,    std     = np.mean(data_norm), np.std(data_norm)
                minimum, maximum = np.min(data_norm),  np.max(data_norm)
                del data_norm

            # whether to make maps with the mean value in all pixels
            if just_monopole:
                data_c = 10**(data_c)
                mean_each_map = np.mean(data_c, axis=(1,2))
                for i in range(data_c.shape[0]):
                    data_c[i] = mean_each_map[i]
                data_c = np.log10(data_c)

            #data = 2*(data - minimum)/(maximum - minimum) - 1.0
            data_c = (data_c - mean)/std
            if verbose:  
                print('%.3f < F(all|norm) < %.3f'%(np.min(data_c), np.max(data_c))) 

            # keep only the data of the chosen set
            data_c = data_c[indexes]

            # do a loop over all rotations (each is 90 deg)
            counted_maps = 0
            for rot in [0,1,2,3]:
                data_rot = np.rot90(data_c, k=rot, axes=(1,2))

                data[counted_maps:counted_maps+size_maps,channel,:,:] = data_rot
                params[counted_maps:counted_maps+size_maps]           = params_maps
                counted_maps += size_maps

                data[counted_maps:counted_maps+size_maps,channel,:,:] = \
                                                    np.flip(data_rot, axis=1)
                params[counted_maps:counted_maps+size_maps]           = params_maps
                counted_maps += size_maps
            
            if verbose:
                print('Channel %d contains %d maps'%(channel,counted_maps))
                print('%.3f < F < %.3f\n'%(np.min(data_c), np.max(data_c)))
                
        
        self.size = data.shape[0]
        self.x    = torch.tensor(data,   dtype=torch.float32)
        self.y    = torch.tensor(params, dtype=torch.float32)
        del data, data_c

        #vutils.save_image(self.x, 'images.png', nrow=10, normalize=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        shape = (self.x[idx]).shape
        indexes = np.arange(torch.numel(self.x[idx]))
        np.random.shuffle(indexes)
        shuffled_map = self.x[idx].view(-1)
        shuffled_map = shuffled_map[indexes]
        shuffled_map = shuffled_map.view(shape)
        return shuffled_map, self.y[idx]
        """
        return self.x[idx], self.y[idx]



# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, f_images, f_params, splits, f_images_norm, 
                 monopole, verbose):

        # read the data
        print('Creating %s dataset'%mode)
        data        = np.load(f_images)    #[number of maps, height, width]
        params_orig = np.loadtxt(f_params) #parameters of simulations, not of maps
        params      = np.zeros((params_orig.shape[0]*splits,6), dtype=np.float64)
        for i in range(params_orig.shape[0]):
            for j in range(splits):
                params[i*splits + j] = params_orig[i]

        # check that numbers are correct
        total_maps, total_sims = data.shape[0], params_orig.shape[0]
        if data.shape[0]!=params.shape[0]:  raise Exception('sizes do not match')

        # normalize maps
        if verbose:  print('%.3e < T(all|orig) < %.3e'%(np.min(data), np.max(data)))
        if f_images.find('Mstar')!=-1:  data = np.log10(data + 1.0)
        else:                           data = np.log10(data)

        # remove monopole of the images
        if monopole is False:
            print('removing monopoles')
            data_mean = np.mean(data, axis=(1,2), dtype=np.float64)
            for i in range(data.shape[0]):
                data[i] = data[i] - data_mean[i]

        if f_images_norm is None:  
            mean,    std     = np.mean(data), np.std(data)
            minimum, maximum = np.min(data),  np.max(data)
        else:
            data_norm        = np.load(f_images_norm)
            if f_images_norm.find('Mstar')!=-1:  data_norm = np.log10(data_norm + 1.0)
            else:                                data_norm = np.log10(data_norm)
            mean,    std     = np.mean(data_norm), np.std(data_norm)
            minimum, maximum = np.min(data_norm),  np.max(data_norm)
            del data_norm
        if verbose:  print('%.3f < T(all|log)  < %.3f'%(np.min(data), np.max(data))) 
        #data = 2*(data - minimum)/(maximum - minimum) - 1.0
        data = (data - mean)/std
        if verbose:  print('%.3f < T(all|norm) < %.3f'%(np.min(data), np.max(data))) 

        # normalize params
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params  = (params - minimum)/(maximum - minimum)

        # get the size and offset depending on the type of dataset
        if   mode=='train':  
            offset, size_sims = int(0.00*total_sims), int(0.90*total_sims)
        elif mode=='valid':  
            offset, size_sims = int(0.90*total_sims), int(0.05*total_sims)
        elif mode=='test':  
            offset, size_sims = int(0.95*total_sims), int(0.05*total_sims)
        elif mode=='all':  
            offset, size_sims = int(0.00*total_sims), int(1.00*total_sims)
        else:    raise Exception('Wrong name!')
        size_maps = size_sims*splits

        # randomly shuffle the simulations (not maps). Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        sim_numbers = np.arange(total_sims) #shuffle maps not rotations
        np.random.shuffle(sim_numbers)
        sim_numbers = sim_numbers[offset:offset+size_sims] #select indexes of mode

        # get the corresponding indexes of the maps associated to the sims
        indexes = np.zeros(size_maps, dtype=np.int32)
        count = 0
        for i in sim_numbers:
            for j in range(splits):
                indexes[count] = i*splits + j
                count += 1

        # keep only the data of the chosen set
        data   = data[indexes]
        params = params[indexes]

        # define the matrix hosting all data with all rotations/flipping
        # together with the array containing the numbers of each map
        data_all    = np.zeros((size_maps*8, data.shape[1], data.shape[2]), 
                               dtype=np.float32)
        params_all  = np.zeros((size_maps*8, params.shape[1]), dtype=np.float32)

        # do a loop over all rotations (each is 90 deg)
        total_maps = 0
        for rot in [0,1,2,3]:
            data_rot = np.rot90(data, k=rot, axes=(1,2))

            data_all[total_maps:total_maps+size_maps,:,:] = data_rot
            params_all[total_maps:total_maps+size_maps]   = params
            total_maps += size_maps

            data_all[total_maps:total_maps+size_maps,:,:] = np.flip(data_rot, axis=1)
            params_all[total_maps:total_maps+size_maps]   = params
            total_maps += size_maps
            
        if verbose:
            print('This set contains %d maps'%total_maps)
            print('%.3f < T (this set) < %.3f\n'%(np.min(data), np.max(data)))

        self.size = data_all.shape[0]
        self.x    = torch.unsqueeze(torch.tensor(data_all, dtype=torch.float32),1)
        self.y    = torch.tensor(params_all, dtype=torch.float32)

        #vutils.save_image(self.x, 'images.png', nrow=10, normalize=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


"""
# This class creates the dataset 
class make_dataset_mixed():

    def __init__(self, mode, seed, f_images, f_params, splits, f_images_norm, 
                 monopole, verbose):

        # read the data
        print('Creating %s dataset'%mode)
        data        = np.load(f_images)    #[number of maps, height, width]
        params_orig = np.loadtxt(f_params) #parameters of simulations, not of maps
        params      = np.zeros((params_orig.shape[0]*splits,6), dtype=np.float64)
        for i in range(params_orig.shape[0]):
            for j in range(splits):
                params[i*splits + j] = params_orig[i]
        if data.shape[0]!=params.shape[0]:  raise Exception('sizes do not match')

        # normalize maps
        if verbose:  print('%.3e < T(all|orig) < %.3e'%(np.min(data), np.max(data)))
        data = np.log10(data)
        if f_images_norm is None:  
            mean,    std     = np.mean(data), np.std(data)
            minimum, maximum = np.min(data),  np.max(data)
        else:
            data_norm        = np.load(f_images_norm)
            data_norm        = np.log10(data_norm)
            mean,    std     = np.mean(data_norm), np.std(data_norm)
            minimum, maximum = np.min(data_norm),  np.max(data_norm)
            del data_norm
        if verbose:  print('%.3f < T(all|log)  < %.3f'%(np.min(data), np.max(data))) 
        #data = 2*(data - minimum)/(maximum - minimum) - 1.0
        data = (data - mean)/std
        if verbose:  print('%.3f < T(all|norm) < %.3f'%(np.min(data), np.max(data))) 

        # normalize params
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params  = (params - minimum)/(maximum - minimum)

        # get the size and offset depending on the type of dataset
        # reserve the maps from the 50 first sims for pure test (test2)
        unique_maps = data.shape[0]
        train_maps, test_maps = unique_maps - 50*splits, 50*splits
        if    mode=='test2':  size = test_maps
        elif  mode=='all':    size = unique_maps
        elif  mode=='train':  
            offset, size = int(train_maps*0.00), int(train_maps*0.70)
        elif mode=='valid':  
            offset, size = int(train_maps*0.70), int(train_maps*0.15)
        elif mode=='test':   
            offset, size = int(train_maps*0.85), int(train_maps*0.15)
        else:    raise Exception('Wrong name!')

        # mix maps for training, validation and testing
        if   mode=='test2':  indexes = np.arange(test_maps)
        elif mode=='all':    indexes = np.arange(unique_maps)
        else:
            # randomly shuffle the maps. Instead of 0 1 2 3...999 have a 
            # random permutation. E.g. 5 9 0 29...342
            np.random.seed(seed)
            indexes = np.arange(test_maps, unique_maps) #shuffle maps not rotations
            np.random.shuffle(indexes)
            indexes = indexes[offset:offset+size] #select indexes of mode

        # keep only the data of the chosen set
        data   = data[indexes]
        params = params[indexes]

        # define the matrix hosting all data with all rotations/flipping
        # together with the array containing the numbers of each map
        data_all    = np.zeros((size*8, data.shape[1], data.shape[2]), dtype=np.float32)
        params_all  = np.zeros((size*8, params.shape[1]),              dtype=np.float32)

        # do a loop over all rotations (each is 90 deg)
        total_maps = 0
        for rot in [0,1,2,3]:
            data_rot = np.rot90(data, k=rot, axes=(1,2))

            data_all[total_maps:total_maps+size,:,:] = data_rot
            params_all[total_maps:total_maps+size]   = params
            total_maps += size

            data_all[total_maps:total_maps+size,:,:] = np.flip(data_rot, axis=1)
            params_all[total_maps:total_maps+size]   = params
            total_maps += size
            
        if verbose:
            print('This set contains %d maps'%total_maps)
            print('%.3f < T (this set) < %.3f\n'%(np.min(data), np.max(data)))

        self.size = data_all.shape[0]
        self.x    = torch.unsqueeze(torch.tensor(data_all, dtype=torch.float32),1)
        self.y    = torch.tensor(params_all, dtype=torch.float32)

        #vutils.save_image(self.x, 'images.png', nrow=10, normalize=True)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
"""


# This routine creates the dataset
# mode ----------> 'train', 'valid', 'test', 'test2', 'all'
# seed ----------> seed to randomly mix the maps into train|valid|test
# f_images ------> name of file with the images
# f_params ------> name of file with value of the simulation (not maps) parameters
# batch_size ----> batch size
# splits --------> total number of maps in a single simulation
# f_images_norm -> images used to normalize the data (e.g. to compute mean and std)
# monopole ------> Whether to remove the mean of each image or not
# verbose -------> prints information about the progress
def create_dataset(mode, seed, f_images, f_params, batch_size, splits, 
                   f_images_norm=None, monopole=True, verbose=False):
    data_set    = make_dataset(mode, seed, f_images, f_params, splits, f_images_norm, 
                               monopole, verbose)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    return data_loader

# This routine creates the dataset for the multifield analysis
# mode ------------> 'train', 'valid', 'test', 'test2', 'all'
# seed ------------> seed to randomly mix the maps into train|valid|test
# f_images --------> tuple containing the names of the fields to consider, e.g.
#                  > ['Images_IllustrisTNG_P_LH_z=0.00.npy',
#                  >  'Images_IllustrisTNG_T_LH_z=0.00.npy']
# f_params --------> name of file with value of the simulation (not maps) parameters
# batch_size ------> batch size
# splits ----------> total number of maps in a single simulation
# f_images_norm ---> tuple containing the names of the images used to normalize the data
#                  > (e.g. to compute mean and std), e.g.:
#                  > ['Images_IllustrisTNG_Mcdm_LH_z=0.00.npy',
#                  >  'Images_IllustrisTNG_Mgas_LH_z=0.00.npy']
#                  > If this data is the same as f_images, then set
#                  > f_images_norm = [None, None]
# num_workers -----> number of cores use to load the data
# monopole --------> Whether to remove the mean of each image or not
# monopole_norm ---> Whether to remove the mean of each image in the normalization set
# rot_flip_in_mem -> whether the dataset contains all rotations and flippings or not
#                  > if True, it will be faster and more efficient, but will use more
#                  > memory. If False, the rotations and flippings are randomly done
#                  > when calling the routine. For validation and testing always set it
#                  > to True. When combining different fields as channels, set it to 
#                  > False or it will not fit in memory.
# shuffle ---------> whether randomly shuffle the data
# just_monopole ---> whether redo images just containing the monopole (only for testing)
# smoothing -------> smooth images with Gaussian kernel; integer with number of pixels
# smoothing_norm --> the smoothing level of the maps used for the normalization
# verbose ---------> prints information about the progress
def create_dataset_multifield(mode, seed, f_images, f_params, batch_size, splits, 
                              f_images_norm, num_workers=1, monopole=True, 
                              monopole_norm=True, rot_flip_in_mem=True, shuffle=True, 
                              just_monopole=False, smoothing=0, smoothing_norm=0, 
                              verbose=False):

    # whether rotations and flippings are kept in memory
    if rot_flip_in_mem:
        data_set = make_dataset_multifield(mode, seed, f_images, f_params, splits, 
                                           f_images_norm, monopole, monopole_norm, 
                                           just_monopole, smoothing, smoothing_norm, 
                                           verbose)
    else:
        data_set = make_dataset_multifield2(mode, seed, f_images, f_params, splits, 
                                            f_images_norm, monopole, monopole_norm, 
                                            smoothing, smoothing_norm, verbose)

    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader

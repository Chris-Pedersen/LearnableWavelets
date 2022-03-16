import torch 
import torch.nn as nn
import numpy as np

""" Full set of CAMELs models imported from 
https://camels-multifield-dataset.readthedocs.io/en/latest/inference.html#scripts
"""


# This function returns the architecture of the considered model
def get_architecture(arch, hidden, dr, channels=1):

    if   arch=='a':       return model_a(hidden)
    elif arch=='b':       return model_b(hidden)
    elif arch=='c':       return model_c(hidden)
    elif arch=='d':       return model_d(hidden)
    elif arch=='e':       return model_e(hidden, dr)
    elif arch=='e2':      return model_e2(hidden, dr)
    elif arch=='e3':      return model_e3(hidden, dr, channels)
    elif arch=='e3_err':  return model_e3_err(hidden, dr, channels)
    elif arch=='e3_abs':  return model_e3_abs(hidden, dr, channels)
    elif arch=='e3_res':  return model_e3_res(hidden, dr)
    elif arch=='f':       return model_f(hidden, dr)
    elif arch=='f2':      return model_f2(hidden, dr)
    elif arch=='f3':      return model_f3(hidden, dr)
    elif arch=='g':       return model_g(hidden, dr)
    elif arch=='g2':      return model_g2(hidden, dr)
    elif arch=='h3_err':  return model_h3_err(hidden, dr, channels)
    elif arch=='i3_err':  return model_i3_err(hidden, dr, channels)
    elif arch=='j3_err':  return model_j3_err(hidden, dr, channels)
    elif arch=='k3_err':  return model_k3_err(hidden, dr, channels)
    elif arch=='l3_err':  return model_l3_err(hidden, dr, channels)
    elif arch=='m3_err':  return model_m3_err(hidden, dr, channels)
    elif arch=='n3_err':  return model_n3_err(hidden, dr, channels)
    elif arch=='o3_err':  return model_o3_err(hidden, dr, channels)
    else:                 raise Exception('Architecture %s not found'%arch)

#####################################################################################
#####################################################################################
class model_a(nn.Module):
    def __init__(self, hidden):
        super(model_a, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx125x125 ----------> output: 2*hiddenx62x62
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=5, stride=2, padding=1,
                            bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx62x62 --------> output: 4*hiddenx31x31
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx31x31 ----------> output: 8*hiddenx15x15
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 8*hiddenx15x15 ----------> output: 16*hiddenx7x7
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 16*hiddenx7x7 ----------> output: 100x3x3
        self.C6 = nn.Conv2d(16*hidden, 100, kernel_size=5, stride=2, padding=1,
                            bias=True)
        self.B6 = nn.BatchNorm2d(100)

        self.FC1  = nn.Linear(100*3*3, 6)  

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_b(nn.Module):
    def __init__(self, hidden):
        super(model_b, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C2 = nn.Conv2d(hidden,    hidden, kernel_size=5, stride=2, padding=1,
                            bias=True)
        self.B2 = nn.BatchNorm2d(hidden)
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C3 = nn.Conv2d(hidden, 2*hidden, kernel_size=4, stride=2, padding=1,
                            bias=True)
        self.B3 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C4 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            bias=True)
        self.B4 = nn.BatchNorm2d(2*hidden)
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C5 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=5, stride=2, padding=1,
                            bias=True)
        self.B5 = nn.BatchNorm2d(4*hidden)
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C6 = nn.Conv2d(4*hidden, 100, kernel_size=5, stride=2, padding=1,
                            bias=True)
        self.B6 = nn.BatchNorm2d(100)

        self.FC1  = nn.Linear(100*3*3, 6)  

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_c(nn.Module):
    def __init__(self, hidden):
        super(model_c, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C6 = nn.Conv2d(16*hidden, 100, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B6 = nn.BatchNorm2d(100)

        self.FC1  = nn.Linear(100*3*3, 6)  

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_d(nn.Module):
    def __init__(self, hidden):
        super(model_d, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C6 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B6 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C1(image))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.FC1(x)

        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_e(nn.Module):
    def __init__(self, hidden, dr):
        super(model_e, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C6 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B6 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.B1(self.C1(image)))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_e2(nn.Module):
    def __init__(self, hidden, dr):
        super(model_e2, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C2 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        self.B2 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C3 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C4 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B3 = nn.BatchNorm2d(2*hidden)
        self.B4 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C5 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C6 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B5 = nn.BatchNorm2d(4*hidden)
        self.B6 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C7 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C8 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B7 = nn.BatchNorm2d(8*hidden)
        self.B8 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C9  = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C10 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B9  = nn.BatchNorm2d(16*hidden)
        self.B10 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C11 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(32*hidden)
        self.B12 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.B1(self.C1(image)))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = self.LeakyReLU(self.B7(self.C7(x)))
        x = self.LeakyReLU(self.B8(self.C8(x)))
        x = self.LeakyReLU(self.B9(self.C9(x)))
        x = self.LeakyReLU(self.B10(self.C10(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_e3(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_e3, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C01 = nn.Conv2d(channels,  hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_e3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_e3_err, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C01 = nn.Conv2d(channels,  hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3, 12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_e3_abs(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_e3_abs, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C01 = nn.Conv2d(channels,  hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = torch.abs(self.FC2(x))
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_e3_res(nn.Module):
    def __init__(self, hidden, dr):
        super(model_e3_res, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C01 = nn.Conv2d(1,         hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)

        self.R02 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)

        self.R12 = nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)

        self.R22 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)

        self.R32 = nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)

        self.R42 = nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        y = self.B02(self.C02(x))
        x = self.LeakyReLU(y)
        x = self.LeakyReLU(self.B03(self.C03(x))) + self.R02(y)
        x = self.LeakyReLU(self.B11(self.C11(x)))
        y = self.B12(self.C12(x))
        x = self.LeakyReLU(y)
        x = self.LeakyReLU(self.B13(self.C13(x))) + self.R12(y)
        x = self.LeakyReLU(self.B21(self.C21(x)))
        y = self.B22(self.C22(x))
        x = self.LeakyReLU(y)
        x = self.LeakyReLU(self.B23(self.C23(x))) + self.R22(y)
        x = self.LeakyReLU(self.B31(self.C31(x)))
        y = self.B32(self.C32(x))
        x = self.LeakyReLU(y)
        x = self.LeakyReLU(self.B33(self.C33(x))) + self.R32(y)
        x = self.LeakyReLU(self.B41(self.C41(x)))
        y = self.B42(self.C42(x))
        x = self.LeakyReLU(y)
        x = self.LeakyReLU(self.B43(self.C43(x))) + self.R42(y)
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_f(nn.Module):
    def __init__(self, hidden, dr):
        super(model_f, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C6 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B6 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3, 2*hidden*3*3)  
        self.FC3  = nn.Linear(2*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.B1(self.C1(image)))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.LeakyReLU(self.FC2(x))
        x = self.FC3(x)
        return x
####################################################################################
####################################################################################


#####################################################################################
#####################################################################################
class model_f2(nn.Module):
    def __init__(self, hidden, dr):
        super(model_f2, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C2 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        self.B2 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C3 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C4 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B3 = nn.BatchNorm2d(2*hidden)
        self.B4 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C5 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C6 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B5 = nn.BatchNorm2d(4*hidden)
        self.B6 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C7 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C8 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B7 = nn.BatchNorm2d(8*hidden)
        self.B8 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C9  = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C10 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B9  = nn.BatchNorm2d(16*hidden)
        self.B10 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C11 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(32*hidden)
        self.B12 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3,  2*hidden*3*3)  
        self.FC3  = nn.Linear(2*hidden*3*3,  6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.B1(self.C1(image)))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = self.LeakyReLU(self.B7(self.C7(x)))
        x = self.LeakyReLU(self.B8(self.C8(x)))
        x = self.LeakyReLU(self.B9(self.C9(x)))
        x = self.LeakyReLU(self.B10(self.C10(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.dropout(self.LeakyReLU(self.FC2(x)))
        x = self.FC3(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_f3(nn.Module):
    def __init__(self, hidden, dr):
        super(model_f3, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C01 = nn.Conv2d(1,         hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 4*hidden*3*3)  
        self.FC2  = nn.Linear(4*hidden*3*3,  2*hidden*3*3)  
        self.FC3  = nn.Linear(2*hidden*3*3,  6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.dropout(self.LeakyReLU(self.FC2(x)))
        x = self.FC3(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_g(nn.Module):
    def __init__(self, hidden, dr):
        super(model_g, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C2 = nn.Conv2d(hidden,   2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B2 = nn.BatchNorm2d(2*hidden)
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C3 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B3 = nn.BatchNorm2d(4*hidden)
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C4 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B4 = nn.BatchNorm2d(8*hidden)
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C5 = nn.Conv2d(8*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B5 = nn.BatchNorm2d(16*hidden)
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C6 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B6 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 16*hidden*3*3)  
        self.FC2  = nn.Linear(16*hidden*3*3, 8*hidden*3*3)  
        self.FC3  = nn.Linear(8*hidden*3*3,  4*hidden*3*3)  
        self.FC4  = nn.Linear(4*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.B1(self.C1(image)))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.dropout(self.LeakyReLU(self.FC2(x)))
        x = self.dropout(self.LeakyReLU(self.FC3(x)))
        x = self.FC4(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_g2(nn.Module):
    def __init__(self, hidden, dr):
        super(model_g2, self).__init__()

        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C1 = nn.Conv2d(1,         hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C2 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B1 = nn.BatchNorm2d(hidden)
        self.B2 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C3 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C4 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B3 = nn.BatchNorm2d(2*hidden)
        self.B4 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C5 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C6 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B5 = nn.BatchNorm2d(4*hidden)
        self.B6 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C7 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C8 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B7 = nn.BatchNorm2d(8*hidden)
        self.B8 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C9  = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C10 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B9  = nn.BatchNorm2d(16*hidden)
        self.B10 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C11 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(32*hidden)
        self.B12 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 16*hidden*3*3)  
        self.FC2  = nn.Linear(16*hidden*3*3, 8*hidden*3*3)  
        self.FC3  = nn.Linear(8*hidden*3*3,  4*hidden*3*3)  
        self.FC4  = nn.Linear(4*hidden*3*3, 6)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.B1(self.C1(image)))
        x = self.LeakyReLU(self.B2(self.C2(x)))
        x = self.LeakyReLU(self.B3(self.C3(x)))
        x = self.LeakyReLU(self.B4(self.C4(x)))
        x = self.LeakyReLU(self.B5(self.C5(x)))
        x = self.LeakyReLU(self.B6(self.C6(x)))
        x = self.LeakyReLU(self.B7(self.C7(x)))
        x = self.LeakyReLU(self.B8(self.C8(x)))
        x = self.LeakyReLU(self.B9(self.C9(x)))
        x = self.LeakyReLU(self.B10(self.C10(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.dropout(self.LeakyReLU(self.FC2(x)))
        x = self.dropout(self.LeakyReLU(self.FC3(x)))
        x = self.FC4(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_h3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_h3_err, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C01 = nn.Conv2d(channels,  hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 16*hidden*3*3)  
        self.FC2  = nn.Linear(16*hidden*3*3,  8*hidden*3*3)  
        self.FC3  = nn.Linear(8*hidden*3*3,   4*hidden*3*3)  
        self.FC4  = nn.Linear(4*hidden*3*3,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.C53(x))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.dropout(self.LeakyReLU(self.FC2(x)))
        x = self.dropout(self.LeakyReLU(self.FC3(x)))
        x = self.FC4(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_i3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_i3_err, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C01 = nn.Conv2d(channels,  hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: hiddenx62x62
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: hiddenx62x62 --------> output: 2*hiddenx31x31
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 2*hiddenx31x31 ----------> output: 2*hiddenx15x15
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 2*hiddenx15x15 ----------> output: 4*hiddenx7x7
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 4*hiddenx7x7 ----------> output: 100x3x3
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)

        self.FC1  = nn.Linear(32*hidden*3*3, 8*hidden*3*3)  
        self.FC2  = nn.Linear(8*hidden*3*3,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.C53(x))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_j3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_j3_err, self).__init__()
        
        # input: 1x250x250 ---------------> output: hiddenx125x125
        self.C01 = nn.Conv2d(channels,  hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=4, stride=2, padding=1, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx125x125 ----------> output: 2*hiddenx62x62
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx62x62 --------> output: 4*hiddenx31x31
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=4, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx31x31 ----------> output: 8*hiddenx15x15
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx15x15 ----------> output: 16*hiddenx7x7
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx7x7 ----------> output: 32*hiddenx3x3
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=5, stride=2, padding=1,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        # input: 32*hiddenx3x3 ----------> output: 64*hiddenx1x1
        self.C61 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C62 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C63 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=5, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(64*hidden)
        self.B62 = nn.BatchNorm2d(64*hidden)
        self.B63 = nn.BatchNorm2d(64*hidden)

        self.FC1  = nn.Linear(64*hidden*1*1,  64*hidden*1*1)  
        self.FC2  = nn.Linear(64*hidden*1*1,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))
        x = self.LeakyReLU(self.B61(self.C61(x)))
        x = self.LeakyReLU(self.B62(self.C62(x)))
        x = self.LeakyReLU(self.C63(x))
        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)
        return x
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_k3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_k3_err, self).__init__()
        
        # input: 1x256x256 ---------------> output: hiddenx128x128
        self.C00 = nn.Conv2d(channels,  hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C01 = nn.Conv2d(channels,  hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx128x128 ----------> output: 2*hiddenx64x64
        self.C10 = nn.Conv2d(hidden,   2*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx64x64 --------> output: 4*hiddenx32x32
        self.C20 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx32x32 ----------> output: 8*hiddenx16x16
        self.C30 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx16x16 ----------> output: 16*hiddenx8x8
        self.C40 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx8x8 ----------> output: 32*hiddenx4x4
        self.C50 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        # input: 32*hiddenx4x4 ----------> output: 64*hiddenx1x1
        self.C60 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C61 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C62 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C63 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(64*hidden)
        self.B62 = nn.BatchNorm2d(64*hidden)
        self.B63 = nn.BatchNorm2d(64*hidden)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.P1  = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)

        self.FC1  = nn.Linear(64*hidden,  64*hidden)  
        self.FC2  = nn.Linear(64*hidden,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        y = self.P0(self.C00(image))
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)) + y)

        y = self.P0(self.C10(x))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)) + y)

        y = self.P0(self.C20(x))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)) + y)

        y = self.P0(self.C30(x))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)) + y)

        y = self.P0(self.C40(x))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)) + y)

        y = self.P0(self.C50(x))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)) + y)

        y = self.P1(self.C60(x))
        x = self.LeakyReLU(self.B61(self.C61(x)))
        x = self.LeakyReLU(self.B62(self.C62(x)))
        x = self.LeakyReLU(self.C63(x) + y)

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,6:12] = torch.square(x[:,6:12])

        return y
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_l3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_l3_err, self).__init__()
        
        # input: 1x256x256 ---------------> output: hiddenx128x128
        self.C00 = nn.Conv2d(channels,  hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C01 = nn.Conv2d(channels,  hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(hidden,    hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(hidden,    hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(hidden)
        self.B02 = nn.BatchNorm2d(hidden)
        self.B03 = nn.BatchNorm2d(hidden)
        
        # input: hiddenx128x128 ----------> output: 2*hiddenx64x64
        self.C10 = nn.Conv2d(hidden,   2*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C11 = nn.Conv2d(hidden,   2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(2*hidden, 2*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(2*hidden)
        self.B12 = nn.BatchNorm2d(2*hidden)
        self.B13 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx64x64 --------> output: 4*hiddenx32x32
        self.C20 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C21 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(4*hidden)
        self.B22 = nn.BatchNorm2d(4*hidden)
        self.B23 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx32x32 ----------> output: 8*hiddenx16x16
        self.C30 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C31 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(8*hidden)
        self.B32 = nn.BatchNorm2d(8*hidden)
        self.B33 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx16x16 ----------> output: 16*hiddenx8x8
        self.C40 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C41 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(16*hidden)
        self.B42 = nn.BatchNorm2d(16*hidden)
        self.B43 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx8x8 ----------> output: 32*hiddenx4x4
        self.C50 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C51 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(32*hidden)
        self.B52 = nn.BatchNorm2d(32*hidden)
        self.B53 = nn.BatchNorm2d(32*hidden)

        # input: 32*hiddenx4x4 ----------> output: 150*hiddenx1x1
        self.C61 = nn.Conv2d(32*hidden, 150*hidden, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(150*hidden, 75*hidden)  
        self.FC2  = nn.Linear(75*hidden,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        y = self.P0(self.C00(image))
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)) + y)

        y = self.P0(self.C10(x))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)) + y)

        y = self.P0(self.C20(x))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)) + y)

        y = self.P0(self.C30(x))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)) + y)

        y = self.P0(self.C40(x))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)) + y)

        y = self.P0(self.C50(x))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)) + y)

        x = self.LeakyReLU(self.C61(x))

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,6:12] = torch.square(x[:,6:12])

        return y
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_m3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_m3_err, self).__init__()
        
        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C00 = nn.Conv2d(channels,  2*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(2*hidden)
        self.B02 = nn.BatchNorm2d(2*hidden)
        self.B03 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C10 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(4*hidden)
        self.B12 = nn.BatchNorm2d(4*hidden)
        self.B13 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C20 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C30 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C40 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(32*hidden)
        self.B42 = nn.BatchNorm2d(32*hidden)
        self.B43 = nn.BatchNorm2d(32*hidden)
        
        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C50 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(64*hidden)
        self.B52 = nn.BatchNorm2d(64*hidden)
        self.B53 = nn.BatchNorm2d(64*hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(128*hidden)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(128*hidden, 64*hidden)  
        self.FC2  = nn.Linear(64*hidden,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        y = self.P0(self.C00(image))
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)) + y)

        y = self.P0(self.C10(x))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)) + y)

        y = self.P0(self.C20(x))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)) + y)

        y = self.P0(self.C30(x))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)) + y)

        y = self.P0(self.C40(x))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)) + y)

        y = self.P0(self.C50(x))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)) + y)

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,6:12] = torch.square(x[:,6:12])

        return y
####################################################################################
####################################################################################


#####################################################################################
#####################################################################################
class model_n3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_n3_err, self).__init__()
        
        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C00 = nn.Conv2d(channels,  2*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(2*hidden)
        self.B02 = nn.BatchNorm2d(2*hidden)
        self.B03 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C10 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(4*hidden)
        self.B12 = nn.BatchNorm2d(4*hidden)
        self.B13 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C20 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C30 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C40 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(32*hidden)
        self.B42 = nn.BatchNorm2d(32*hidden)
        self.B43 = nn.BatchNorm2d(32*hidden)
        
        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C50 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=1, stride=1, padding=0, 
                             padding_mode='circular', bias=True)
        self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(64*hidden)
        self.B52 = nn.BatchNorm2d(64*hidden)
        self.B53 = nn.BatchNorm2d(64*hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(128*hidden)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(128*hidden, 128*hidden)  
        self.FC2  = nn.Linear(128*hidden,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        y = self.P0(self.C00(image))
        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)) + y)

        y = self.P0(self.C10(x))
        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)) + y)

        y = self.P0(self.C20(x))
        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)) + y)

        y = self.P0(self.C30(x))
        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)) + y)

        y = self.P0(self.C40(x))
        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)) + y)

        y = self.P0(self.C50(x))
        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)) + y)

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,6:12] = torch.square(x[:,6:12])

        return y
####################################################################################
####################################################################################

#####################################################################################
#####################################################################################
class model_o3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_o3_err, self).__init__()
        
        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(2*hidden)
        self.B02 = nn.BatchNorm2d(2*hidden)
        self.B03 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(4*hidden)
        self.B12 = nn.BatchNorm2d(4*hidden)
        self.B13 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(32*hidden)
        self.B42 = nn.BatchNorm2d(32*hidden)
        self.B43 = nn.BatchNorm2d(32*hidden)
        
        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(64*hidden)
        self.B52 = nn.BatchNorm2d(64*hidden)
        self.B53 = nn.BatchNorm2d(64*hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=4, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(128*hidden)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(128*hidden, 64*hidden)  
        self.FC2  = nn.Linear(64*hidden,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))

        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))

        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))

        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))

        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,6:12] = torch.square(x[:,6:12])

        return y
####################################################################################
####################################################################################

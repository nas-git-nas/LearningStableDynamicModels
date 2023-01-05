import torch
import torch.nn as nn
import numpy as np

class ModelBlackSimple(nn.Module):
    def __init__(self, args, dev):
        """
        Args:
            args: argument class instance
            dev: pytorch device
        """
        super(ModelBlackSimple, self).__init__()

        # system parameters
        self.D = 6 # dim. of state x
        self.M = 6 # dim. of controll input u   
        self.S = 3 # dim. of space

        hidden_size = 64
        nb_hidden_layers = 5

        nnx_input_size = [self.M + 1]
        nny_input_size = [self.M + 1]
        nnt_input_size = [self.M]
        nnx_input_size.extend([hidden_size]*nb_hidden_layers)
        nny_input_size.extend([hidden_size]*nb_hidden_layers) 
        nnt_input_size.extend([hidden_size]*nb_hidden_layers)

        output_size = [hidden_size]*nb_hidden_layers
        output_size.append(1)
        

        self.nnx_lin_fcts = nn.ModuleList([nn.Linear(input, output, bias=True, dtype=torch.float32) 
                                            for (input, output) in zip(nnx_input_size,output_size)])
        self.nny_lin_fcts = nn.ModuleList([nn.Linear(input, output, bias=True, dtype=torch.float32) 
                                            for (input, output) in zip(nny_input_size,output_size)])
        self.nnt_lin_fcts = nn.ModuleList([nn.Linear(input, output, bias=True, dtype=torch.float32) 
                                            for (input, output) in zip(nnt_input_size,output_size)])

        self.tanh = nn.Tanh()

    def forward(self, X, U):
        """
        Forward pass through main model
        Args:
            X: state input batch (N, D)
            U: controll input batch (N, M)
        Returns:
            dX_X: state derivative (N, D)
        """
        acc = torch.zeros(X.shape[0],self.S)
        acc[:,0] = self.forwardAcc(Y=torch.concat([U,X[:,2,np.newaxis]], axis=1), lin_fcts=self.nnx_lin_fcts)
        acc[:,1] = self.forwardAcc(Y=torch.concat([U,X[:,2,np.newaxis]], axis=1), lin_fcts=self.nny_lin_fcts)
        acc[:,2] = self.forwardAcc(Y=U, lin_fcts=self.nnt_lin_fcts)
        
        dX_X = torch.concat((X[:,3:6], acc), axis=1)       
        return dX_X


    def forwardAcc(self, Y, lin_fcts):
        """
        Correct acceleration of grey box model
        Args:
            Y: control input concatenated with theta, tensor (N,M) or (N,M+1)
            lin_fcts: list of linear functions to apply, nn.ModuleList
        """
        for lin in lin_fcts[0:-1]:
            Y = lin(Y)
            Y = self.tanh(Y)
        
        return lin_fcts[-1](Y).flatten()
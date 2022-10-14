import numpy as np
import torch

class ContinuousStirredTankReactorGenerator():
    def __init__(self, dev, controlled_system=False):

        self.device = dev
        self.controlled_system=controlled_system

        # system parameters
        self.D = 2 # number of state dimensions
        self.M = 1 # nb. of control dimensions
        self.mass = 1
        self.phi1 = 0.0041
        self.phi2 = 0.0041
        self.phi3 = 0.00063
        self.cA0 = 5.1

        # system boundaries
        self.x_min = 1
        self.x_max = 3
        self.dx_min = 0.5
        self.dx_max = 2
        self.u_min = 3
        self.u_max = 35

        # data
        self.X = None
        self.U = None
        self.dX_X = None

    def generate_data(self, N, nb_batches):
        """
        Description: generate one batch of samples X and its derivative f_X
        In N: batch size
        In nb_batches: number of batches
        Out X: sample data (nb_batches x N x D)
        Out dX_X: derivative of X (nb_batches x N x D)
        """
        self.X = torch.zeros((nb_batches,N,self.D), dtype=float)
        self.U = torch.zeros((nb_batches,N,self.M), dtype=float)
        self.dX_X = torch.zeros((nb_batches,N,self.D), dtype=float)
        for i in range(nb_batches):
            self.X[i,:,:], self.U[i,:,:] = self.generate_input(N)
            self.dX_X[i,:,:] = self.calc_dX_X(self.X[i,:,:], self.U[i,:,:])

        self.X = self.X.float().to(self.device)
        self.U = self.U.float().to(self.device)
        self.dX = self.dX_X.float().to(self.device)
    
    def getData(self):
        return self.X.detach().clone(), self.U.detach().clone(), self.dX.detach().clone()
                

    def generate_input(self, N):
        """
        Description: generate one batch of samples X
        In N: batch size
        Out X: sample data (N x D)
        Out U: controll input data (N x M)
        """        
        X = torch.rand(N, self.D)
        X[:,0] = X[:,0]*(self.x_max-self.x_min) + self.x_min
        X[:,1] = X[:,1]*(self.dx_max-self.dx_min) + self.dx_min

        U = torch.empty((N, self.M))
        if self.controlled_system:
            U = torch.rand(N, self.M)
            U = U*(self.u_max-self.u_min) + self.u_min
       
        return X, U

    def calc_dX_X(self, X, U):
        """
        Description: calc. derivative of damped harmonic oscillator
        In X: batch of sample data (N x D)
        In U: controll input data (N x M)
        Out dX: derivative of X (N x D)
        """
        dX = torch.empty(X.shape)
        dX[:,0] = -self.phi1*X[:,0] - self.phi3*torch.pow(X[:,0], 2)
        dX[:,1] = self.phi1*X[:,0] - self.phi2*torch.pow(X[:,1], 2)

        if self.controlled_system:
            dX[:,0] = dX[:,0] + U[:,0]*(self.cA0 - X[:,0])
            dX[:,1] += -U[:,0]*X[:,1]

        return dX


def test_oscillator():
    # parameters
    batch_size = 3
    gen = ContinuousStirredTankReactorGenerator()

    # generate batch
    X, U, f_X = gen.generate_data(batch_size)
    print(f"X = {X}, \nU = {U}, \nf_X = {f_X}")

if __name__ == "__main__":
    test_oscillator()
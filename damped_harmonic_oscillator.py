import numpy as np
import torch

class DampedHarmonicOscillator():
    def __init__(self, dev):

        self.device = dev

        # system parameters
        self.D = 2 # number of state dimensions
        self.M = 1 # nb. of control dimensions
        self.mass = 1
        self.spring_const = 0.5
        self.friction_coeff = 0.1

        # system boundaries
        self.x_min = -0.5
        self.x_max = 1.5
        self.dx_min = -0.5
        self.dx_max = 0.5
        self.u_min = -1
        self.u_max = 1

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
        self.X = np.zeros((nb_batches,N,self.D), dtype=float)
        self.U = np.zeros((nb_batches,N,self.M), dtype=float)
        self.dX_X = np.zeros((nb_batches,N,self.D), dtype=float)
        for i in range(nb_batches):
            self.X[i,:,:], self.U[i,:,:] = self.generate_input(N)
            self.dX_X[i,:,:] = self.calc_dX_X(self.X[i,:,:], self.U[i,:,:])

        self.X = torch.from_numpy(self.X).float().to(self.device)
        self.U = torch.from_numpy(self.U).float().to(self.device)
        self.dX = torch.from_numpy(self.dX_X).float().to(self.device)
    
    def getData(self):
        return self.X.detach().clone(), self.U.detach().clone(), self.dX.detach().clone()
                

    def generate_input(self, N):
        """
        Description: generate one batch of samples X
        In N: batch size
        Out X: sample data (N x D)
        """        
        X = np.random.rand(N, self.D)
        X[:,0] = X[:,0]*(self.x_max-self.x_min) + self.x_min
        X[:,1] = X[:,1]*(self.dx_max-self.dx_min) + self.dx_min

        U = np.random.rand(N, self.M)
        U = U*(self.u_max-self.u_min) + self.u_min
       
        return X, U

    def calc_dX_X(self, X, U):
        """
        Description: calc. derivative of damped harmonic oscillator
        In X: batch of sample data (N x D)
        Out dX: derivative of X (N x D)
        """
        A = np.array([[0, 1],[-self.spring_const/self.mass, -self.friction_coeff/self.mass]])
        B = np.array([[0], [-1/self.mass]])

        dX = X@(A.T) + U@(B.T)
        return dX


def test_oscillator():
    # parameters
    batch_size = 3
    dho = DampedHarmonicOscillator()

    # generate batch
    X, U, f_X = dho.generate_data(batch_size)
    print(f"X = {X}, \nU = {U}, \nf_X = {f_X}")

if __name__ == "__main__":
    test_oscillator()
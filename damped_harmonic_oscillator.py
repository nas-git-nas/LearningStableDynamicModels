import numpy as np
import torch

class DampedHarmonicOscillator():
    def __init__(self, controlled_system, dev):

        self.device = dev
        self.controlled_system = controlled_system

        # system parameters
        self.D = 2 # number of state dimensions
        self.M = 1 # nb. of control dimensions
        self.mass = 1
        self.spring_const = 0.5
        self.friction_coeff = 0.1

        # system boundaries
        self.x_min = -1
        self.x_max = 1
        self.dx_min = -0.1
        self.dx_max = 0.1
        self.u_min = -1
        self.u_max = 1

    def generate_data(self, N, nb_batches):
        """
        Description: generate one batch of samples X and its derivative f_X
        In N: batch size
        In nb_batches: number of batches
        Out X: sample data (N x D)
        Out f_X: derivative of X (N x D)
        """
        for _ in range(nb_batches):
            X, U = self.generate_input(N)
            dX_X = self.dX_X(X, U)
            if self.controlled_system:
                yield torch.from_numpy(X).float().to(self.device), torch.from_numpy(U).float().to(self.device), torch.from_numpy(dX_X).float().to(self.device)
            else:
                yield torch.from_numpy(X).float().to(self.device), None, torch.from_numpy(dX_X).float().to(self.device)

    def generate_input(self, N):
        """
        Description: generate one batch of samples X
        In N: batch size
        Out X: sample data (N x D)
        """        
        X = np.random.rand(N, self.D)
        X[:,0] = X[:,0]*(self.x_max-self.x_min) + self.x_min
        X[:,1] = X[:,1]*(self.dx_max-self.dx_min) + self.dx_min

        U = None
        if self.controlled_system:
            U = np.random.rand(N, self.M)
            U = U*(self.u_max-self.u_min) + self.u_min
       
        return X, U

    def dX_X(self, X, U):
        """
        Description: calc. derivative of damped harmonic oscillator
        In X: batch of sample data (N x D)
        Out f_X: derivative of X (N x D)
        """
        A = np.array([[0, 1],[-self.spring_const/self.mass, -self.friction_coeff/self.mass]])
        B = np.array([[0], [-1/self.mass]])

        dX = X@(A.T)
        if self.controlled_system:
            dX += U@(B.T)

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
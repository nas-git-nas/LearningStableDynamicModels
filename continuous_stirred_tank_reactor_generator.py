from unittest import skip
import numpy as np
import torch

class ContinuousStirredTankReactorGenerator():
    def __init__(self, dev, controlled_system=False):

        self.device = dev
        self.controlled_system=controlled_system

        # TODO: set seed

        # system parameters
        self.D = 2 # number of state dimensions
        self.M = 1 # nb. of control dimensions
        self.phi1 = 0.0041
        self.phi2 = 0.0041
        self.phi3 = 0.00063
        self.cA0 = 5.1

        # system boundaries
        self.x_min = torch.tensor([1, 0.5])
        self.x_max = torch.tensor([3, 2])
        self.u_min = 3
        self.u_max = 35

        # controll input map from [u_min,u_max] to [-1,1] 
        # where u_hat = u_map_a * u + u_map_b
        self.u_map_a = 2/(self.u_max-self.u_min) # u_hat_max - u_hat_min = 1-(-1) = 2
        self.u_map_b = 1 - self.u_map_a*self.u_max # u_hat_max = 1       

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
    
    def getData(self, u_map=False):
        U = self.U.detach().clone()
        if u_map:
            U = self.uMap(U)
        
        return self.X.detach().clone(), U, self.dX.detach().clone()               

    def generate_input(self, N):
        """
        Description: generate one batch of samples X
        In N: batch size
        Out X: sample data (N x D)
        Out U: controll input data (N x M)
        """        
        
        X = torch.einsum('nd,d->nd', torch.rand(N, self.D), (self.x_max-self.x_min)) + self.x_min
        # X = torch.rand(N, self.D)
        # X[:,0] = X[:,0]*(self.x_max-self.x_min) + self.x_min
        # X[:,1] = X[:,1]*(self.dx_max-self.dx_min) + self.dx_min

        U = torch.empty((N, self.M))
        if self.controlled_system:
            U = torch.rand(N, self.M)
            U = U*(self.u_max-self.u_min) + self.u_min
       
        return X, U

    def calc_dX_X(self, X, U, U_hat=False):
        """
        Description: calc. derivative of damped harmonic oscillator
        In X: batch of sample data (N x D)
        In U: controll input data (N x M)
        Out dX: derivative of X (N x D)
        """
        if U_hat:
            U = self.uMapInv(U)

        dX = torch.empty(X.shape)
        dX[:,0] = -self.phi1*X[:,0] - self.phi3*torch.pow(X[:,0], 2)
        dX[:,1] = self.phi1*X[:,0] - self.phi2*torch.pow(X[:,1], 2)

        # if self.controlled_system:
        dX[:,0] = dX[:,0] + U[:,0]*(self.cA0 - X[:,0])
        dX[:,1] += -U[:,0]*X[:,1]

        return dX

    def uMap(self, U):
        U = U.detach().clone()
        return self.u_map_a*U + self.u_map_b

    def uMapInv(self, U):
        U = U.detach().clone()
        return (U-self.u_map_b)/self.u_map_a   

    def calcEquPoint(self, U, U_hat=False):
        """
        Calc. equilibrium points of system, in general there exist four solutions
        Args:
            U: control input (scalar)
            U_hat: if true then controll input is mapped from [-1,1] to [self.u_min,self.u_max]
        Returns:
            equ_points: tensor with all the equilibrium points (4,2)
            closest_equ_point: tensor with the closest equilibrium point to the range of X (2)
        """
        # convert U_hat to U
        if U_hat:
            U = self.uMapInv(U)

        # calc. four different solutions
        poly_cA = np.polynomial.polynomial.Polynomial([self.cA0*U, -(self.phi1+U), -self.phi3], domain=[1.0,  3.0], window=[1.0,  3.0])
        roots_cA = poly_cA.roots()
        poly_cB1 = np.polynomial.polynomial.Polynomial([self.phi1*roots_cA[0], -U, -self.phi2], domain=[0.5,  2.0], window=[0.5,  2.0])
        roots_cB1 = poly_cB1.roots()
        poly_cB2 = np.polynomial.polynomial.Polynomial([self.phi1*roots_cA[1], -U, -self.phi2], domain=[0.5,  2.0], window=[0.5,  2.0])
        roots_cB2 = poly_cB2.roots()
        equ_points = torch.tensor([[roots_cA[0],roots_cB1[0]],
                                   [roots_cA[0],roots_cB1[1]],
                                   [roots_cA[1],roots_cB2[0]],
                                   [roots_cA[1],roots_cB2[1]]])

        # verify which equilibrium point is the closest to the center
        X_center = (self.x_max-self.x_min)/2 + self.x_min
        best_dist = np.Inf
        closest_equ_point = None       
        for eq in equ_points:
            # skip point if it is a complex solution
            if np.any(np.iscomplex(eq)):
                continue

            # calc. distance to center
            dist = torch.linalg.norm(eq.real - X_center)
            if dist < best_dist:
                best_dist = dist
                closest_equ_point = eq.real

        # verify if derivative of X is zero at equilibrium point
        dX = self.calc_dX_X(X=closest_equ_point.clone().detach().reshape(1,2), U=torch.tensor([[U]]))
        error = torch.linalg.norm(dX)
        assert error < 0.0001, "Closest equilibrium point has large error"

        return equ_points, closest_equ_point


def test_oscillator():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev) 

    # parameters
    gen = ContinuousStirredTankReactorGenerator(dev=device, controlled_system=True)

    # # generate batch
    # X, U, f_X = gen.generate_data(batch_size)
    # print(f"X = {X}, \nU = {U}, \nf_X = {f_X}")
    equ_points, closest_equ_point = gen.calcEquPoint(U=14.19)

    print(f"Closest equ. point is: {closest_equ_point}")


if __name__ == "__main__":
    test_oscillator()
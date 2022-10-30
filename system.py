from abc import abstractmethod
import numpy as np
import torch

class System:
    def __init__(self, dev, controlled_system=True):

        self.device = dev
        self.controlled_system=controlled_system

        # data
        self.X = None
        self.U = None
        self.dX = None

        # system boundaries
        self.x_min = None
        self.x_max = None
        self.u_min = None
        self.u_max = None

    @abstractmethod
    def calcDX(self, X, U, U_hat=False):
        pass

    @abstractmethod
    def uMap(self, U):
        pass

    @abstractmethod
    def uMapInv(self, U):
        pass

    @abstractmethod
    def _calcEquPoints(self, U):
        pass

    @abstractmethod
    def _verifyStability(self, equ_points, U):
        pass

    @abstractmethod
    def _closestEquPoint(self, equ_points, U, stabilities):
        pass

    def getData(self, u_map=False):
        U = self.U.detach().clone()
        if u_map:
            U = self.uMap(U)
        
        return self.X.detach().clone(), U, self.dX.detach().clone()

    def generateData(self, N, nb_batches):
        """
        Description: generate one batch of samples X and its derivative f_X
        In N: batch size
        In nb_batches: number of batches
        Out X: sample data (nb_batches x N x D)
        Out dX_X: derivative of X (nb_batches x N x D)
        """
        self.X = torch.zeros((nb_batches,N,self.D), dtype=float)
        self.U = torch.zeros((nb_batches,N,self.M), dtype=float)
        self.dX = torch.zeros((nb_batches,N,self.D), dtype=float)
        for i in range(nb_batches):
            self.X[i,:,:], self.U[i,:,:] = self.generateX(N)
            self.dX[i,:,:] = self.calcDX(self.X[i,:,:], self.U[i,:,:])

        self.X = self.X.float().to(self.device)
        self.U = self.U.float().to(self.device)
        self.dX = self.dX.float().to(self.device)

    def generateX(self, N):
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

    def equPoint(self, U, U_hat=False):
        """
        Calc. equilibrium points of system an verify if it is stable
        Args:
            U: control input (scalar)
            U_hat: if true then controll input is mapped from [-1,1] to [self.u_min,self.u_max]
        Returns:
            closest_equ_point: tensor with the closest equilibrium point to the range of X (2)
        """
        # convert U_hat to U
        if U_hat:
            U = self.uMapInv(U)

        equ_points = self._calcEquPoints(U)
        stabilities = self._verifyStability(equ_points, U)
        closest_equ_point, is_stable = self._closestEquPoint(equ_points, U, stabilities)

        return closest_equ_point, is_stable




class CSTR(System):
    def __init__(self, dev, controlled_system=True):
        System.__init__(self, dev, controlled_system=True)

        # system parameters
        self.D = 2 # number of state dimensions
        self.M = 1 # nb. of control dimensions

        k10  =  1.287e12
        k20  =  1.287e12
        k30  =  9.043e09
        E1   =  -9758.3
        E2   =  -9758.3
        E3   =  -8560.0
        T    = 1.1419108442079495e02
        self.phi1  = k10*np.exp(E1/(273.15 + T))
        self.phi2  = k20*np.exp(E2/(273.15 + T))
        self.phi3  = k30*np.exp(E3/(273.15 + T))
        self.TIMEUNITS_PER_HOUR = 3600.0
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

    def calcDX(self, X, U, U_hat=False):
        """
        Description: calc. derivative of X
        In X: batch of sample data (N x D)
        In U: controll input data (N x M)
        Out dX: derivative of X (N x D)
        """
        if U_hat:
            U = self.uMapInv(U)

        dX = torch.zeros(X.shape)
        dX[:,0] = (1/self.TIMEUNITS_PER_HOUR)*(U[:,0]*(self.cA0 - X[:,0])-self.phi1*X[:,0]-self.phi3*torch.pow(X[:,0], 2))
        dX[:,1] = (1/self.TIMEUNITS_PER_HOUR)*(-U[:,0]*X[:,1]+self.phi1*X[:,0]-self.phi2*X[:,1])

        return dX

    def uMap(self, U):
        U = U.detach().clone()
        return self.u_map_a*U + self.u_map_b

    def uMapInv(self, U):
        U = U.detach().clone()
        return (U-self.u_map_b)/self.u_map_a

    def _calcEquPoints(self, U):
        """
        Calc. equilibrium points of system, in general there exist four solutions
        Args:
            U: control input (scalar)
        Returns:
            equ_points: tensor with all the equilibrium points (4,2)
        """
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
        return equ_points

    def _verifyStability(self, equ_points, U):
        """
        Verify if point is stable equilibrium, stable iff all eigenvalues of hessian are > 0 (positive semi-definite hessian)
        Args:
            equ_points: tensor with all the equilibrium points (4,2)
            U: control input (scalar)
        Returns:
            stabilities: array indicating if equilibrium point is stable (True) or not (False) (4)
        """
        stabilities = np.ones((4), dtype=bool)
        for i, point in enumerate(equ_points):
            # calc. eigenvalues of hessian
            hess = np.array([[(1/self.TIMEUNITS_PER_HOUR)*(-U-self.phi1-2*self.phi3*point[0]), 0],
                            [(1/self.TIMEUNITS_PER_HOUR)*(self.phi1), (1/self.TIMEUNITS_PER_HOUR)*(-U-self.phi2)]])
            _lambdas, _ = np.linalg.eig(hess)

            # verify if all eigenvalues are strictly greater than zero
            for _lambda in _lambdas:
                if _lambda.real <= 0:
                    stabilities[i] = False

        return stabilities
    
    def _closestEquPoint(self, equ_points, U, stabilities):
        """
        Calc. closest equilibrium point
        Args:
            equ_points: tensor with all the equilibrium points (4,2)
            U: control input (scalar)
            stabilities: array indicating if equilibrium point is stable (True) or not (False) (4)
        Returns:           
            closest_equ_point: tensor with the closest equilibrium point to the range of X (2)
            is_stable: indicating if closest equilibrium point is stable
        """
        # verify which equilibrium point is the closest to the center
        X_center = (self.x_max-self.x_min)/2 + self.x_min
        best_dist = np.Inf
        closest_equ_point = []
        is_stable = None       
        for i, point in enumerate(equ_points):
            # skip point if it is a complex solution
            if np.any(np.iscomplex(point)):
                continue

            # calc. distance to center
            dist = torch.linalg.norm(point.real - X_center)
            if dist < best_dist:
                best_dist = dist
                closest_equ_point = point.real
                is_stable = stabilities[i]

        # verify if stable equilibrium point exists
        assert len(closest_equ_point) > 0, "No real solution was found"

        # verify if derivative of X is zero at equilibrium point
        dX = self.calcDX(X=closest_equ_point.clone().detach().reshape(1,2), U=torch.tensor([[U]]))
        assert torch.linalg.norm(dX) < 0.1, "Closest equilibrium point has large error"

        return closest_equ_point, is_stable



def test_oscillator():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev) 

    # parameters
    gen = CSTR(dev=device, controlled_system=True)

    # # generate batch
    # gen.generateData(N=20, nb_batches=1)
    # X, U, f_X = gen.getData()
    # print(f"X = {X}, \nU = {U}, \nf_X = {f_X}")

    closest_equ_point, is_stable = gen.equPoint(U=14.19)
    print(f"Closest equ. point: {closest_equ_point}, is stabel: {is_stable}")


if __name__ == "__main__":
    test_oscillator()
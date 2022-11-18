from abc import abstractmethod
import os
import numpy as np
import torch

class System:
    def __init__(self, args, dev):

        self.args = args
        self.device = dev

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
    def generateX(self, N):
        pass

    @abstractmethod
    def calcDX(self, X, U, U_hat=False):
        pass

    @abstractmethod
    def equPointCalc(self, U):
        pass

    @abstractmethod
    def hessian(self, x, u):
        pass

    def getData(self, u_map=False):
        U = self.U.detach().clone()
        if u_map:
            U = self.uMap(U)
        
        return self.X.detach().clone(), U, self.dX.detach().clone()

    def loadData(self):
        """
        Description: load data from csv files: X, U and dX
        """
        X = np.genfromtxt(os.path.join("experiment", self.series, "data_state.csv"), delimiter=",")
        U = np.genfromtxt(os.path.join("experiment", self.series, "data_input.csv"), delimiter=",")
        dX = np.genfromtxt(os.path.join("experiment", self.series, "data_dynamics.csv"), delimiter=",")

        self.X = torch.tensor(X).float().to(self.device)
        self.U = torch.tensor(U).float().to(self.device)
        self.dX = torch.tensor(dX).float().to(self.device)

    def generateData(self, nb_data):
        """
        Description: generate data
        Args:
            nb_data: number of data samples to generate
        """
        X = torch.einsum('nd,d->nd', torch.rand(nb_data, self.D), (self.x_max-self.x_min)) + self.x_min

        U = torch.empty((nb_data, self.M))
        if self.args.controlled_system:
            U = torch.einsum('nm,m->nm', torch.rand(nb_data, self.M), (self.u_max-self.u_min)) + self.u_min      

        dX = self.calcDX(X, U)

        self.X = X.float().to(self.device)
        self.U = U.float().to(self.device)
        self.dX = dX.float().to(self.device)
        
    def uMap(self, U):
        """
        Maps control input from u in [u_min,u_max] to uhat in [-1, 1].
        This is necessar because the model requires abs(u)<=1.
        Args:
            U: control input tensor (M) contained in [u_min,u_max]
        Returns:
            Uhat: control input tensor (M) contained in [-1,1]
        """
        U = U.detach().clone()
        Uhat = (U-self.u_min)/(self.u_max-self.u_min) * 2 - 1
        return Uhat

    def uMapInv(self, Uhat):
        """
        Maps control input from uhat in [-1, 1] to u in [u_min,u_max].
        This is necessar because the model requires abs(u)<=1 but systems not.
        Args:
            Uhat: control input tensor (M) contained in [-1,1]
        Returns:
            U: control input tensor (M) contained in [u_min,u_max]
        """
        Uhat = Uhat.detach().clone()
        U = (Uhat+1)/2 * (self.u_max-self.u_min) + self.u_min
        return U

    def equPoint(self, U, U_hat=False):
        """
        Calc. equilibrium points of system an verify if it is stable
        Args:
            U: control input (M)
            U_hat: if true then controll input is mapped from [-1,1] to [self.u_min,self.u_max]
        Returns:
            closest_equ_point: tensor with the closest equilibrium point to the range of X (2)
        """
        # convert U_hat to U
        if U_hat:
            U = self.uMapInv(U)

        equ_points = self.equPointCalc(U)
        stabilities = self.equPointStab(equ_points, U)
        closest_point = self.equPointChoose(equ_points, U, stabilities)

        return closest_point

    def equPointStab(self, equ_points, U):
        """
        Verify if point is stable equilibrium, stable iff all eigenvalues of hessian are > 0 (positive semi-definite hessian)
        Args:
            equ_points: tensor with all the equilibrium points (4,2)
            U: control input (M)
        Returns:
            stabilities: array indicating if equilibrium point is stable (True) or not (False) (4)
        """
        stabilities = np.ones((4), dtype=bool)
        for i, point in enumerate(equ_points):
            # calc. eigenvalues of hessian
            hess = self.hessian(point, U)
            _lambdas, _ = np.linalg.eig(hess)

            # verify if all eigenvalues are strictly greater than zero
            for _lambda in _lambdas:
                if _lambda.real >= 0:
                    stabilities[i] = False

        return stabilities

    def equPointChoose(self, equ_points, U, stabilities):
        """
        Calc. closest equilibrium point
        Args:
            equ_points: tensor with all the equilibrium points (4,2)
            U: control input (scalar)
            stabilities: array indicating if equilibrium point is stable (True) or not (False) (4)
        Returns:           
            best_point: tensor with the closest equilibrium point to the range of X (2)
        """
        # verify which equilibrium point is the closest to the center
        x_center = (self.x_max-self.x_min)/2 + self.x_min
        best_dist = np.Inf
        best_point = None       
        for i, point in enumerate(equ_points):
            # skip point if it is a complex solution
            if np.any(np.iscomplex(point)):
                continue

            # skip point if it is not stable
            if not stabilities[i]:
                continue

            # calc. distance to center
            dist = torch.linalg.norm(point.real - x_center)
            if dist < best_dist:
                best_dist = dist
                best_point = point.real

        # verify if stable equilibrium point exists
        assert len(best_point) > 0, "No stable real solution was found"

        # verify if derivative of X is zero at equilibrium point
        dX = self.calcDX(X=best_point.clone().detach().reshape(1,2), U=torch.tensor([[U]]))
        assert torch.linalg.norm(dX) < 0.1, "Closest equilibrium point has large error"

        return best_point

class HolohoverSystem(System):
    def __init__(self, args, dev):
        System.__init__(self, args, dev)
        # system dimensions
        self.D = 6 # number of state dimensions
        self.M = 6 # nb. of control dimensions

        # system boundaries
        self.x_min = torch.tensor([-0.5, -0.5, -3.2, -0.5, -0.5, -5])
        self.x_max = torch.tensor([0.5, 0.5, 3.2, 0.5, 0.5, 5])
        self.u_min = torch.zeros(self.M)
        self.u_max = torch.ones(self.M)

        # experiment
        self.series = args.series




class CSTRSystem(System):
    def __init__(self, args, dev):
        System.__init__(self, args, dev)
        # system dimensions
        self.D = 2 # number of state dimensions
        self.M = 1 # nb. of control dimensions

        # system parameters
        self.phi1  = 1.287e12*np.exp(-9758.3/(273.15 + 1.1419108e2))
        self.phi2  = 1.287e12*np.exp(-9758.3/(273.15 + 1.1419108e2))
        self.phi3  = 9.043e09*np.exp(-8560.0/(273.15 + 1.1419108e2))
        self.TIMEUNITS_PER_HOUR = 3600.0
        self.cA0 = 5.1

        # system boundaries
        self.x_min = torch.tensor([1, 0.5])
        self.x_max = torch.tensor([3, 2])
        self.u_min = torch.tensor([3])
        self.u_max = torch.tensor([35])

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

    def equPointCalc(self, U):
        """
        Calc. equilibrium points of system, in general there exist four solutions
        Args:
            U: control input (M)
        Returns:
            equ_points: tensor with all the equilibrium points (4,2)
        """
        U = U.clone().detach().numpy()[0]
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

    def hessian(self, X, U):
        """
        Calc. hessian of state (gradient of dynamics)
        Args:
            x: state (D)
            u: control input (M)
        Returns:
            hess: hessian of x
        """
        hessian = np.array([[(1/self.TIMEUNITS_PER_HOUR)*(-U[0]-self.phi1-2*self.phi3*X[0]), 0],
                            [(1/self.TIMEUNITS_PER_HOUR)*(self.phi1), (1/self.TIMEUNITS_PER_HOUR)*(-U[0]-self.phi2)]])
        return hessian


class DHOSystem(System):
    def __init__(self, args, dev):
        System.__init__(self, args, dev)

        # system dimensions
        self.D = 2 # number of state dimensions
        self.M = 1 # nb. of control dimensions

        # system parameters
        self.mass = 1
        self.spring_const = 0.5
        self.friction_coeff = 0.1

        # system boundaries
        self.x_min = torch.tensor([-0.5, -0.5])
        self.x_max = torch.tensor([1.5, 0.5])
        self.u_min = torch.tensor([-1])
        self.u_max = torch.tensor([1])

    def calcDX(self, X, U, U_hat=False):
        """
        Description: calc. derivative of X
        In X: batch of sample data (N x D)
        In U: controll input data (N x M)
        Out dX: derivative of X (N x D)
        """
        if U_hat:
            U = self.uMapInv(U)

        A = np.array([[0, 1],[-self.spring_const/self.mass, -self.friction_coeff/self.mass]])
        B = np.array([[0], [-1/self.mass]])

        dX = X@(A.T) + U@(B.T)
        return dX

    def equPointCalc(self, U):
        """
        Calc. equilibrium points of system, in general there exist four solutions
        Args:
            U: control input (M)
        Returns:
            equ_points: tensor with all the equilibrium points (4,2)
        """
        return torch.tensor([[-U[0]/self.spring_const, 0.0]])

    def hessian(self, X, U):
        """
        Calc. hessian of state (gradient of dynamics)
        Args:
            x: state (D)
            u: control input (M)
        Returns:
            hess: hessian of x
        """
        hess = np.array([[0.0, 1.0],
                         [-self.spring_const/self.mass, -self.friction_coeff/self.mass]])
        return hess


def test_oscillator():
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"
    device = torch.device(dev) 

    sys = CSTRSystem(dev=device, controlled_system=True)

    closest_equ_point = sys.equPoint(U=torch.tensor([14.19]))
    print(f"Closest equ. point: {closest_equ_point}")


if __name__ == "__main__":
    test_oscillator()
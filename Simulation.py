import torch
import numpy as np
import cvxpy as cp

from continuous_stirred_tank_reactor_model import ContinuousStirredTankReactorModel
from system import CSTR

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev)

class Simulation():
    def __init__(self):
        

        self.gen = CSTR(dev=device, controlled_system=self.controlled_system)
        self.model = ContinuousStirredTankReactorModel(controlled_system=self.controlled_system, 
                                                    lyapunov_correction=self.lyapunov_correction, 
                                                    generator=self.gen, dev=device)


        model_path = "\models\continuous_stirred_tank_reactor\20221030_1752"
        self.model.load_state_dict(torch.load(model_path))

    def simulate(self):
        X0 = self.gen.x_min.reshape(1,self.gen.D)
        U0 = torch.tensor(self.gen.uMap(self.gen.u_min)).reshape(1,self.gen.M)
        nb_steps = 10

        X_list = [X0]
        U_list = [U0]
        for i in range(nb_steps):
            X = X_list[-1]
            U =
            U = self.safetyFilter(X[-1], U[-1])
            dX = self.gen.calcDX(X, U, U_hat=True)
            X





    def safetyFilter(self, X, Udis):

        # X (1, D)
        # Udis (1, M)

        # calc. learned system dynamics
        with torch.no_grad():
            f_X = self.model.forwardFNN(X) # (N x D)
            g_X = self.model.forwardGNN(X) # (N x D x M)
            V = self.model.forwardLyapunov(X) # (N)
            dV = self.model.gradient_lyapunov(X) # (N x D)
            f_opt = f_X + self.model.fCorrection(f_X, g_X, V, dV)

        # colapse dimension N because N=1 and conver to numpy
        f_X = f_X[0,:].cpu().detach().numpy() # (D)
        g_X = g_X[0,:,:].cpu().detach().numpy() # (D,M)
        V = V[0].cpu().detach().numpy() # (1)
        dV = dV[0,:].cpu().detach().numpy() # (D)
        f_opt = f_opt[0,:].cpu().detach().numpy() # (D)
        Udis = Udis[0,:].cpu().detach().numpy() # (M)

        # QP constraints
        G = np.concatenate((np.einsum('d,dm->m',dV, g_X).reshape(1,self.gen.M), 
                            np.identity(self.gen.M), 
                            -np.identity(self.gen.M)), axis=0) # (1+2*M,M)
        h = np.concatenate((np.einsum('d,d->',dV, g_X@Udis)-np.einsum('d,d->',dV,f_X)-self.model.alpha*V, 
                            np.ones((self.gen.M))-Udis, 
                            np.ones((self.gen.M))+Udis), axis=0) # (1+2*M)
        A = np.zeros((1,self.gen.M)) # (1,M)
        b = np.zeros((1)) # (1)

        P = np.identity(self.gen.M)
        q = np.zeros((self.gen.M))

        usafe = cp.Variable(self.gen.M)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(usafe, P) + q.T @ usafe), [G @ usafe <= h, A @ usafe == b])
        prob.solve()

        # Print result.
        print("\nThe optimal value is", prob.value)
        print("A solution usafe is")
        print(usafe.value)
        print("A dual solution corresponding to the inequality constraints is")
        print(prob.constraints[0].dual_value)



def testSafetyFilter():
    sim = Simulation()

if __name__ == "__main__":
    testSafetyFilter()
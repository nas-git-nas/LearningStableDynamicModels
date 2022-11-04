import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from system import DHOSystem, CSTRSystem
from model import DHOModel, CSTRModel

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev)

class Simulation():
    def __init__(self):
        # system type
        self.model_type = "DHO"        
        self.controlled_system = True 
        self.lyapunov_correction = True

        # initialize system
        if self.model_type == "DHO":
            self.sys = DHOSystem(dev=device, controlled_system=self.controlled_system)
        elif self.model_type == "CSTR":
            self.sys = CSTRSystem(dev=device, controlled_system=self.controlled_system)

        # calc. equilibrium point
        self.ueq = torch.tensor([0])  #torch.tensor([14.19]) 
        self.xeq = self.sys.equPoint(self.ueq, U_hat=False)
        self.ueq = self.sys.uMap(self.ueq) # ueq_hat -> ueq=14.19

        # init. model
        if self.model_type == "DHO":
            self.model = DHOModel(controlled_system=self.controlled_system, 
                                  lyapunov_correction=self.lyapunov_correction, 
                                  generator=self.sys, dev=device, xref=self.xeq)
        elif self.model_type == "CSTR":       
            self.model = CSTRModel(controlled_system=self.controlled_system, 
                                   lyapunov_correction=self.lyapunov_correction, 
                                   generator=self.sys, dev=device, xref=self.xeq)


        model_path = "models/DHO/20221103_0822/20221103_0822_model"
        self.model.load_state_dict(torch.load(model_path))

    def simulate(self):
        # X0 = self.sys.x_min.reshape(1,self.sys.D)
        # U0 = torch.tensor(self.sys.uMap(self.sys.u_min)).reshape(1,self.sys.M)
        X0 = torch.tensor([0.0, 0.0]).reshape(1,self.sys.D)
        U0 = torch.tensor([2]).reshape(1,self.sys.M)
        nb_steps = 250
        periode = 0.1

        X_list = [X0]
        U_list = [U0]
        for i in range(nb_steps):
            # get current values
            X = X_list[-1]
            U = U_list[-1]
            U = torch.tensor([i**2/400000])

            # calc. safe control input and system dynamics
            U = self.safetyFilter(X, U)
            dX = self.sys.calcDX(X, U, U_hat=True)
            X = X + periode*dX #small timestep

            # append results
            X_list.append(X)
            U_list.append(U)

        fig = plt.figure()
        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        color = plt.cm.rainbow(np.linspace(0, 1, len(X_list)))
        for i, X in enumerate(X_list):
            if not i%50:
                ax1.plot(X[0,0], X[0,1], 'o', color=color[i], label="State "+str(i))
            else:
                ax1.plot(X[0,0], X[0,1], 'o', color=color[i])
            
        ax1.legend()
        print(self.xeq)
        plt.show()


    def safetyFilter(self, X, Udis):


        X = X.detach().clone().tile(2,1).float()
        Udis = Udis.detach().clone().tile(2,1).float()

        # print(f"X = {X}")
        # print(f"Udis = {Udis}")

        # calc. learned system dynamics
        with torch.no_grad():
            f_X = self.model.forwardFNN(X) # (N x D)
            g_X = self.model.forwardGNN(X) # (N x D x M)
            deltaX = X - self.model.Xref.tile(X.shape[0],1)
            V = self.model.forwardLyapunov(deltaX) # (N)
            dV = self.model.gradient_lyapunov(deltaX) # (N x D)
            f_opt = f_X + self.model.fCorrection(f_X, g_X, V, dV)

        # colapse dimension N because N=1 and conver to numpy
        f_X = f_X[0,:].cpu().detach().numpy() # (D)
        g_X = g_X[0,:,:].cpu().detach().numpy() # (D,M)
        V = V[0].cpu().detach().numpy() # (1)
        dV = dV[0,:].cpu().detach().numpy() # (D)
        f_opt = f_opt[0,:].cpu().detach().numpy() # (D)
        Udis = Udis[0,:].cpu().detach().numpy() # (M)

        # QP constraints
        G = np.concatenate((np.einsum('d,dm->m',dV, g_X).reshape(1,self.sys.M), 
                            np.identity(self.sys.M), 
                            -np.identity(self.sys.M)), axis=0) # (1+2*M,M)
        h = np.zeros((1 + 2*self.sys.M)) # (1+2*M)
        h[0] = - np.einsum('d,d->',dV,f_opt) - self.model.alpha*V
        h[1:1+self.sys.M] = np.ones((self.sys.M))
        h[1+self.sys.M:1+2*self.sys.M] = np.ones((self.sys.M))
        # A = np.zeros((1,self.sys.M)) # (1,M)
        # b = np.zeros((1)) # (1)

        P = np.identity(self.sys.M)
        q = Udis
        slack_coeff = 1000

        # print(f"G = {G}")
        # print(f"h = {h}")
        # print(f"P = {P}")
        # print(f"q = {q}")

        usafe = cp.Variable(self.sys.M)
        slack = cp.Variable(1 + 2*self.sys.M)

        obj = cp.Minimize((1/2)*cp.sum_squares(usafe) - q.T@usafe + slack_coeff*cp.sum(slack))
        # obj = cp.Minimize((1/2)*cp.quad_form(usafe, P) - q.T@usafe)
        con = [G @ usafe - slack <= h, slack >= np.zeros((1 + 2*self.sys.M))]
        # con = [G @ usafe <= h]

        # prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(usafe, P) + q.T @ usafe), [G @ usafe <= h, A @ usafe == b])
        prob = cp.Problem(obj, con)
        prob.solve()

        # Print result.
        # print("status:", prob.status)
        # print("\nThe optimal value is:", prob.value)
        # print("A solution usafe is")
        # print("Sol. usafe: {usafe.value}")
        # print("A dual solution corresponding to the inequality constraints is")
        # print(prob.constraints[0].dual_value)

        print(f"Udis: {Udis}, Sol. usafe: {usafe.value}, Sol. slack: {slack.value}")
        # print(f"Sol. usafe: {usafe.value}")
        Usafe = torch.tensor(usafe.value, dtype=float)
        
        return Usafe




def testSafetyFilter():
    sim = Simulation()

    X = torch.tensor([0, 0]).reshape(1,sim.sys.D) #sim.gen.x_min.reshape(1,sim.gen.D)
    U = torch.tensor([1]).reshape(1,sim.sys.M) #sim.gen.uMap(torch.tensor(sim.gen.u_min).reshape(1,sim.gen.M))
    # sim.safetyFilter(X, U)
    sim.simulate()

if __name__ == "__main__":
    testSafetyFilter()
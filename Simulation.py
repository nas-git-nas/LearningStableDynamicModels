import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from system import DHOSystem, CSTRSystem
from model import DHOModel, CSTRModel
from plot import Plot

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

        # simulation params
        self.slack_coeff = 1000

        self.plot = Plot(model=self.model, system=self.sys, dev=device)

    def simulation(self):
        # X0 = self.sys.x_min.reshape(1,self.sys.D)
        # U0 = torch.tensor(self.sys.uMap(self.sys.u_min)).reshape(1,self.sys.M)
        nb_steps = 150
        periode = 0.01
        X0 = np.array([0.0, 0.0])
        # Udes_seq = np.array([i/100 for i in range(nb_steps)]).reshape(nb_steps,self.sys.M)
        Udes_seq = np.array([1 for i in range(nb_steps)]).reshape(nb_steps,self.sys.M)


        X_seq_on, Usafe_seq_on, slack_seq_on = self.simSys(nb_steps, periode, X0, Udes_seq, safety_filter=True)
        X_seq_off, _, _ = self.simSys(nb_steps, periode, X0, Udes_seq, safety_filter=False)

        self.plot.sim(X_seq_on, X_seq_off, Udes_seq, Usafe_seq_on, slack_seq_on)

    def simSys(self, nb_steps, periode, X0, Udes_seq, safety_filter=True):
        """
        Simulate system with a given control input sequence
        Args:
            nb_steps: number of simulation steps
            periode: simulation periode
            X0: starting state at time step 0, numpy array (D)
            Udes_seq: sequence of desired control inputs, numpy array (nb_steps,M)
            safety_filter: if True safety filter is active
        Returns:
            X_seq: resulting state sequence (nb_steps+1,D)
            Usafe_seq: resulting control input sequence (nb_steps,M)
            slack_seq: resulting slacks used in optimization (nb_steps)
        """
        X_seq = np.zeros((nb_steps+1, self.sys.D))
        X_seq[0,:] = X0
        Usafe_seq = np.zeros((nb_steps, self.sys.M))
        slack_seq = np.zeros((nb_steps))

        X = X0
        for i in range(nb_steps):
            # calc. safe control input
            if safety_filter:
                Usafe, slack = self.safetyFilter(X, Udes_seq[i,:])
            else:
                Usafe = Udes_seq[i,:]
                slack = 0
            
            # update state with system dynamicys
            dX = self.sys.calcDX(torch.tensor(X).reshape(1,self.sys.D), 
                                 torch.tensor(Usafe).reshape(1,self.sys.M), U_hat=True)
            X = X + periode*dX.detach().numpy()

            # append results
            X_seq[i+1,:] = X
            Usafe_seq[i,:] = Usafe
            slack_seq[i] = np.sum(slack)

        return X_seq, Usafe_seq, slack_seq

    def safetyFilter(self, X, Udes):
        """
        Calc. safe control input by solving optimization problem
        Args:
            X: state, array (1,D)
            Udes: desired control input, array (1,M)
        Returns:
            Usafe: safe control input, array (M)
            slack: slack necessary to make the optimization feasible, scalar
        """
        # doublicate X and Udes because model requires N>=2
        X = torch.tensor(X).reshape(1,self.sys.D).tile(2,1).float()
        Udes = torch.tensor(Udes).reshape(1,self.sys.M).tile(2,1).float()

        # calc. learned system dynamics
        with torch.no_grad():
            f_X = self.model.forwardFNN(X) # (N x D)
            g_X = self.model.forwardGNN(X) # (N x D x M)
            deltaX = X - self.model.Xref.tile(X.shape[0],1)
            V = self.model.forwardLyapunov(deltaX) # (N)
            dV = self.model.gradient_lyapunov(deltaX) # (N x D)
            f_opt = f_X + self.model.fCorrection(f_X, g_X, V, dV)

        # colapse dimension N because N=1 and convert tensor to array
        f_X = f_X[0,:].cpu().detach().numpy() # (D)
        g_X = g_X[0,:,:].cpu().detach().numpy() # (D,M)
        V = V[0].cpu().detach().numpy() # (1)
        dV = dV[0,:].cpu().detach().numpy() # (D)
        f_opt = f_opt[0,:].cpu().detach().numpy() # (D)
        Udes = Udes[0,:].cpu().detach().numpy() # (M)   

        # define optimization variables
        usafe = cp.Variable(self.sys.M)
        slack = cp.Variable(1 + 2*self.sys.M)

        # define objective and constraints and solve optimization problem
        obj = cp.Minimize((1/2)*cp.sum_squares(usafe) - Udes.T@usafe + self.slack_coeff*cp.sum(slack))
        con = [np.einsum('d,dm->m',dV, g_X).T@usafe + np.einsum('d,d->',dV,f_opt) + self.model.alpha*V <= slack[0], 
               usafe <= np.ones((self.sys.M)) + slack[1:1+self.sys.M],
               -usafe <= np.ones((self.sys.M)) + slack[1+self.sys.M:],
               np.zeros((1 + 2*self.sys.M)) <= slack]
        prob = cp.Problem(obj, con)
        prob.solve()
        
        return np.array(usafe.value).reshape(self.sys.M), slack.value



def testSafetyFilter():
    sim = Simulation()

    X = torch.tensor([0, 0]).reshape(1,sim.sys.D) #sim.gen.x_min.reshape(1,sim.gen.D)
    U = torch.tensor([1]).reshape(1,sim.sys.M) #sim.gen.uMap(torch.tensor(sim.gen.u_min).reshape(1,sim.gen.M))
    # sim.safetyFilter(X, U)
    sim.simulation()

if __name__ == "__main__":
    testSafetyFilter()
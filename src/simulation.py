import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev)

class Simulation():
    def __init__(self, sys, model):
        
        self.sys = sys
        self.model = model  

        # simulation params
        self.slack_coeff = 1000

    def simGrey(self):
        nb_steps = 1200
        periode = 0.01
        
        X, U_seq, dX_seq = self.sys.getData(u_map=True)

        Xreal_seq = self.simRealSys(nb_steps, periode, X[0,:], dX_seq)
        Xlearn_seq, _, _, _ = self.simLearnedSys(   nb_steps, periode, X[0,:].detach().numpy(), 
                                                    U_seq.detach().numpy(), self.model.forward)
        
        return Xreal_seq, Xlearn_seq


    def simRealSys(self, nb_steps, periode, X0, dX_seq): 

        X_seq = np.zeros((nb_steps+1, self.sys.D))
        X_seq[0,:] = X0

        X = X0
        for i in range(nb_steps):           
            # update state with system dynamicys
            dX = dX_seq[i,:]
            X = X + periode*dX.detach().numpy()

            # append results
            X_seq[i+1,:] = X

        return X_seq
    
    def simLearnedSys(self, nb_steps, periode, X0, Udes_seq, dX_fct, safety_filter=False):
        """
        Simulate system with a given control input sequence
        Args:
            nb_steps: number of simulation steps
            periode: simulation periode
            X0: starting state at time step 0, numpy array (D)
            Udes_seq: sequence of desired control inputs, numpy array (nb_steps,M)
            dX_fct: function to calc. derivative of X, function with args (X, U)
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
        V_seq = np.zeros((nb_steps))

        X = X0
        for i in range(nb_steps):
            # calc. safe control input
            if safety_filter:
                Usafe, slack, V = self.safetyFilter(X, Udes_seq[i,:])
            else:
                Usafe = Udes_seq[i,:]
                slack = 0
                V = 0
            
            # update state with system dynamicys
            dX = dX_fct(torch.tensor(X).reshape(1,self.sys.D), 
                        torch.tensor(Usafe).reshape(1,self.sys.M))
            X = X + periode*dX.detach().numpy()

            # append results
            X_seq[i+1,:] = X
            Usafe_seq[i,:] = Usafe
            slack_seq[i] = np.sum(slack)
            V_seq[i] = V

        return X_seq, Usafe_seq, slack_seq, V_seq

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
        
        return np.array(usafe.value).reshape(self.sys.M), slack.value, V



def testSafetyFilter():
    sim = Simulation()

    X = torch.tensor([0, 0]).reshape(1,sim.sys.D) #sim.gen.x_min.reshape(1,sim.gen.D)
    U = torch.tensor([1]).reshape(1,sim.sys.M) #sim.gen.uMap(torch.tensor(sim.gen.u_min).reshape(1,sim.gen.M))
    # sim.safetyFilter(X, U)
    sim.simulation()

if __name__ == "__main__":
    testSafetyFilter()
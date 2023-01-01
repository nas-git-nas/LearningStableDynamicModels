import os
import torch
import numpy as np
import cvxpy as cp

class Simulation():
    def __init__(self, sys, model):
        """
        Args:
            sys: system class instance
            model: model class instance
        """        
        self.sys = sys
        self.model = model

        # simulation params
        self.slack_coeff = 1000

    def simGrey(self):
        """
        Simulate grey model
        """
        series = "validation_20221208"
        
        X = np.genfromtxt(os.path.join("experiment", series, "data_state.csv"), delimiter=",")
        U = np.genfromtxt(os.path.join("experiment", series, "data_input.csv"), delimiter=",")
        dX = np.genfromtxt(os.path.join("experiment", series, "data_dynamics.csv"), delimiter=",")
        tX = np.genfromtxt(os.path.join("experiment", series, "data_time.csv"), delimiter=",")

        nb_steps = tX.shape[0] - 1
        Xreal_seq = X
        Upoly = self.sys.polyExpandU(U=torch.tensor(U)).detach().numpy()
        Xlearn_seq, _, _, _ = self.simLearnedSys(nb_steps=nb_steps, tX=tX, X0=X[0,:], Udes_seq=Upoly, dX_fct=self.model.forward)
        Xreal_integ_seq = self.simRealSys(nb_steps=nb_steps, tX=tX, X0=X[0,:], dX_seq=dX)
        
        return Xreal_seq, Xreal_integ_seq, Xlearn_seq


    def simRealSys(self, nb_steps, X0, dX_seq, periode=None, tX=None):
        """
        Simulate system using real acceleration
        Args:
            nb_steps: number of simulation steps
            X0: starting state at time step 0, numpy array (D)
            dX_seq: sequence of real system dynamics, numpy array (nb_steps,D)
            periode: simulation periode
            tX: time sequence of control inputs, will be used instead of periode
        Returns:
            X_seq: resulting state sequence (nb_steps+1,D)
        """
        X_seq = np.zeros((nb_steps+1, self.sys.D))
        X_seq[0,:] = X0

        X = X0
        for i in range(nb_steps):
            # calc. periode if time sequence is provided
            if len(tX) > 0:
                periode = tX[i+1] - tX[i]
                   
            # update state with system dynamicys
            dX = dX_seq[i,:]
            X = X + periode*dX

            # append results
            X_seq[i+1,:] = X

        return X_seq
    
    def simLearnedSys(self, nb_steps, X0, Udes_seq, dX_fct, safety_filter=False, periode=None, tX=None):
        """
        Simulate system with a given control input sequence
        Args:
            nb_steps: number of simulation steps
            X0: starting state at time step 0, numpy array (D)
            Udes_seq: sequence of desired control inputs, numpy array (nb_steps,M)
            dX_fct: function to calc. derivative of X, function with args (X, U)
            safety_filter: if True safety filter is active
            periode: simulation periode
            tX: time sequence of control inputs, will be used instead of periode
        Returns:
            X_seq: resulting state sequence (nb_steps+1,D)
            Usafe_seq: resulting control input sequence (nb_steps,M)
            slack_seq: resulting slacks used in optimization (nb_steps)
        """
        X_seq = np.zeros((nb_steps+1, self.sys.D))
        X_seq[0,:] = X0
        Usafe_seq = np.zeros((nb_steps, Udes_seq.shape[1]))
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

            # calc. periode if time sequence is provided
            if len(tX) > 0:
                periode = tX[i+1] - tX[i]
            
            # update state with system dynamicys
            dX = dX_fct(torch.tensor(X).float().reshape(1,self.sys.D), 
                        torch.tensor(Usafe).float().reshape(1,len(Usafe)))
            dX = dX.detach().numpy().flatten()
            X = X + periode*dX

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

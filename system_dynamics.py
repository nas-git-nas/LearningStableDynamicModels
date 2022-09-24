import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from damped_harmonic_oscillator import DampedHarmonicOscillator
from fcnn import FCNN
from icnn import ICNN

class SystemDynamics():
    def __init__(self):
        # neural network parameters
        icnn_learning_rate = 0.01
        self.nb_epochs = 10000
        self.batch_size = 10

        # dynamic model parameters
        self.alpha = 0.1 # constant ???
        self.loss_constant = 1

        # real data model
        self.dho = DampedHarmonicOscillator()

        self.fcnn = FCNN()
        self.icnn = ICNN()
        self.optimizer = torch.optim.SGD(self.icnn.parameters(), lr=icnn_learning_rate)

    def optimize(self):
        """
        Description: run nb_epochs optimizations
        """
        losses = []
        for _ in range(self.nb_epochs):
            # generate batch of data
            X, f_real = self.dho.generate_batch(self.batch_size)

            # forward pass through models
            f_X = self.fcnn.forward(X) # output of FCNN if input X (n)
            with torch.no_grad():
                g_zero = self.icnn_zero.forward(torch.zeros(X.shape[1])) # output of ICNN if input x=0 (scalar)
            g_X = self.icnn.forward(X) # output of ICNN if input X (b)

            # calc. loss
            loss = self.loss_function(f_X, f_real, g_X, g_zero)
            losses.append(loss.item())

            # backwards pass through models
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def loss_function(self, f_X, f_real, g_X, g_zero):
        """
        Description: calc. average loss of batch X
        In f_X: approx. of system dynamics by FCNN (b x n)
        In f_real: real system dynamics (b x n)
        In g_X: output of ICNN if input X (b)
        In g_zero: output of ICNN if input x=0 (scalar)
        Out L: average loss of batch X
        """
        V = self.icnn.lyapunov(X, g_X, g_zero) # (b)
        dV = self.icnn.gradient_lyapunov(X, g_X) # (b x n)

        f_opt = f_X
        stability_conditions = np.diagonal(dV@f_X.T) + self.alpha*V # (b)
        for i, sc in enumerate(stability_conditions):
            if sc > 0: # if it is necessary correct f to ensure stability
                f_opt[i,:] -= sc*(dV[i,:]/np.sum(dV[i,:]*dV[i,:]))


        return (self.loss_constant/f_opt.shape[0]) * np.sum(np.square(f_opt-f_real))



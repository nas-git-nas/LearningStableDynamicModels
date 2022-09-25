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
        self.fcnn_learning_rate = 0.01
        self.nb_epochs = 1000
        self.batch_size = 100
        self.nb_batches = 100

        # dynamic model parameters
        self.alpha = 0.1 # constant ???
        self.loss_constant = 1

        # real data model
        self.dho = DampedHarmonicOscillator()

        self.fcnn = FCNN()
        # self.icnn = ICNN()
        self.optimizer = torch.optim.SGD(self.fcnn.parameters(), lr=self.fcnn_learning_rate)
        self.loss_fct = nn.MSELoss()
        self.losses = []

    # def optimize(self):
    #     """
    #     Description: run nb_epochs optimizations
    #     """
    #     for _ in range(self.nb_epochs):
    #         # generate batch of data
    #         X, f_real = self.dho.generate_batch(self.batch_size)

    #         # forward pass through models
    #         f_X = self.fcnn.forward(X) # output of FCNN if input X (n)
    #         with torch.no_grad():
    #             g_zero = self.icnn.forward(torch.zeros(X.shape[1])) # output of ICNN if input x=0 (scalar)
    #         g_X = self.icnn.forward(X) # output of ICNN if input X (b)

    #         # calc. loss
    #         loss = self.loss_function(X, f_X, f_real, g_X, g_zero)
    #         self.losses.append(loss.item())

    #         # backwards pass through models
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    # def loss_function(self, X, f_X, f_real, g_X, g_zero):
    #     """
    #     Description: calc. average loss of batch X
    #     In f_X: approx. of system dynamics by FCNN (b x n)
    #     In f_real: real system dynamics (b x n)
    #     In g_X: output of ICNN if input X (b)
    #     In g_zero: output of ICNN if input x=0 (scalar)
    #     Out L: average loss of batch X
    #     """
    #     V = self.icnn.lyapunov(X, g_X, g_zero) # (b)
    #     dV = self.icnn.gradient_lyapunov(X, g_X) # (b x n)

    #     # f_opt = f_X
    #     # stability_conditions = torch.diagonal(dV@f_X.T) + self.alpha*V # (b)
    #     # print(stability_conditions)
    #     # for i, sc in enumerate(stability_conditions):
    #     #     if sc > 0: # if it is necessary correct f to ensure stability
    #     #         f_opt[i,:] -= sc*(dV[i,:]/np.sum(dV[i,:]*dV[i,:]))

    #     stability_conditions = torch.diagonal(dV@f_X.T) + self.alpha*V # (b)
    #     relu = nn.ReLU()
    #     dV_norm = (dV.T//torch.sum(dV*dV, dim=1)).T
    #     f_opt = f_X - (dV_norm.T*relu(stability_conditions)).T

    #     return (self.loss_constant/f_opt.shape[0]) * torch.sum(torch.square(f_opt-f_real))

    def optimize_fcnn(self):
        """
        Description: run nb_epochs optimizations
        """
        # generate data set
        data_X, data_real = self.dho.generate_batch(self.batch_size*self.nb_batches)

        for _ in range(self.nb_epochs):
            for i in range(self.nb_batches):
                X = data_X[i:(i+1)*self.batch_size,:]
                f_real = data_real[i:(i+1)*self.batch_size,:]

                # forward pass through models
                f_X = self.fcnn.forward(X) # output of FCNN if input X (n)

                # calc. loss
                loss = self.loss_function_fcnn(f_X, f_real)
                self.losses.append(loss.item())

                # backwards pass through models
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def loss_function_fcnn(self, f_X, f_real):
        """
        Description: calc. average loss of batch X
        In f_X: approx. of system dynamics by FCNN (b x n)
        In f_real: real system dynamics (b x n)
        Out L: average loss of batch X (scalar)
        """
        return (self.loss_constant/f_X.shape[0]) * torch.sum(torch.square(f_X-f_real))

    def plot_losses(self):
        # test model on test data set
        test_x, test_y = self.dho.generate_batch(self.batch_size)
        pred_y = self.fcnn.forward(test_x)
        loss = self.loss_function_fcnn(pred_y, test_y)
        print(f"Error on test data set is = {loss}")

        for param in self.fcnn.parameters():
            print(param.data)

        # plot resulting loss
        plt.plot(self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("Learning rate %f"%(self.fcnn_learning_rate))
        plt.show()


def learnSystemDynamics():
    sd = SystemDynamics()
    sd.optimize_fcnn()
    sd.plot_losses()

if __name__ == "__main__":
    learnSystemDynamics()      



import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from damped_harmonic_oscillator import DampedHarmonicOscillator
from sdnn import SDNN

class LearnDynamics():
    def __init__(self):
        # neural network parameters
        self.learning_rate = 0.01
        self.nb_epochs = 1000
        self.batch_size = 100
        self.nb_batches = 100

        # dynamic model parameters
        self.alpha = 0.1 # constant ???
        self.loss_constant = 1

        # real data model
        self.dho = DampedHarmonicOscillator()

        self.sdnn = SDNN()
        self.optimizer = torch.optim.SGD(self.sdnn.parameters(), lr=self.learning_rate)
        self.losses = []

    def optimize(self):
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
                print(f"X shape = {X.shape}")
                f_X = self.sdnn.forward(X) # output of FCNN if input X (n)

                # calc. loss
                loss = self.loss_function(f_X, f_real)
                self.losses.append(loss.item())

                # backwards pass through models
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def loss_function(self, f_X, f_real):
        """
        Description: calc. average loss of batch X
        In f_X: approx. of system dynamics by FCNN (b x n)
        In f_real: real system dynamics (b x n)
        Out L: average loss of batch X (scalar)
        """
        return (self.loss_constant/f_X.shape[0]) * torch.sum(torch.square(f_X-f_real))

    def plot_losses(self):
        # test model on test data set
        test_X, test_real = self.dho.generate_batch(self.batch_size)
        test_f_X = self.sdnn.forward(test_X)
        loss = self.loss_function(test_f_X, test_real)
        print(f"Error on test data set is = {loss}")

        # for param in self.fcnn.parameters():
        #     print(param.data)

        # plot resulting loss
        plt.plot(self.losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("Learning rate %f"%(self.fcnn_learning_rate))
        plt.show()


def learnSystemDynamics():
    ld = LearnDynamics()
    ld.optimize()
    ld.plot_losses()

if __name__ == "__main__":
    learnSystemDynamics()      



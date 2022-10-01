import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from damped_harmonic_oscillator import DampedHarmonicOscillator
from csdnn import CSDNN

class LearnDynamics():
    def __init__(self):
        # neural network parameters
        self.learning_rate = 0.01
        self.nb_epochs = 100
        self.batch_size = 100
        self.nb_batches = 100

        # dynamic model parameters
        self.alpha = 0.1 # constant ???
        self.loss_constant = 1
        self.controlled_system = False
        self.lyapunov_correction = False

        # real dynamic system
        self.dho = DampedHarmonicOscillator(controlled_system=self.controlled_system)

        # modelled dynamic system
        self.sdnn = CSDNN(controlled_system=self.controlled_system, lyapunov_correction=self.lyapunov_correction)
        self.optimizer = torch.optim.SGD(self.sdnn.parameters(), lr=self.learning_rate)
        self.loss_batches = np.zeros((self.nb_epochs*self.nb_batches))
        self.loss_epochs = np.zeros((self.nb_epochs))

    def optimize(self):
        """
        Description: optimize nb_epochs times the model with all data (batch_size*nb_batches)
        """
        # generate data set
        # data_X, data_U, data_real = self.dho.generate_data(self.batch_size*self.nb_batches) # TODO: generator function
        data = self.dho.generate_data(self.batch_size, self.nb_batches)

        start_time = time.time()
        for j in range(self.nb_epochs):
            for i, batch_data in enumerate(data):
                # X = data_X[i*self.batch_size:(i+1)*self.batch_size,:]
                # U = None
                # if self.controlled_system:
                #     U = data_U[i*self.batch_size:(i+1)*self.batch_size,:]
                # dX_real = data_real[i*self.batch_size:(i+1)*self.batch_size,:]
                X, U, dX_real = batch_data[0], batch_data[1], batch_data[2]

                # forward pass through models
                dX_X = self.sdnn.forward(X, U) # output of FCNN if input X (n)

                # calc. loss
                loss = self.loss_function(dX_X, dX_real)
                self.loss_batches[self.nb_batches*j+i] = loss.item()

                # backwards pass through models
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.loss_epochs[j] = np.mean(self.loss_batches[self.nb_batches*j:self.nb_batches*(j+1)])
            print(f"Epoch {j}: Avg. loss = {self.loss_epochs[j]}, lr = {self.learning_rate}")

            # delta_loss = np.abs(self.loss_epochs[j]-self.loss_epochs[j-1]) # TODO: step wise learning rate
            # if delta_loss < 0.01:
            #     self.learning_rate *= 0.8
            # elif delta_loss > 0.1:
            #     self.learning_rate *= 1.2


            
        end_time =time.time()
        print(f"\nTotal time = {end_time-start_time}, average time per epoch = {(end_time-start_time)/self.nb_epochs}")

    def loss_function(self, dX_X, dX_real):
        """
        Description: calc. average loss of batch X
        In dX_X: approx. of system dynamics (b x n)
        In dX_real: real system dynamics (b x n)
        Out L: average loss of batch X (scalar)
        """
        return (self.loss_constant/dX_X.shape[0]) * torch.sum(torch.square(dX_X-dX_real))

    def plot_losses(self):
        """
        Description: plot losses and print some weights"""

        # TODO: plot lyapunov fct. and norm2 of lyapunov ||V||2

        # test model on test data set
        test_X, test_U, test_real = self.dho.generate_data(self.batch_size)
        test_dX_X = self.sdnn.forward(test_X, test_U)
        loss = self.loss_function(test_dX_X, test_real)
        print(f"Error = {loss}")

        # for name, para in self.sdnn.named_parameters():
        #     print('{}: {}, \n{}'.format(name, para.shape, para))

        fnn_weights = self.sdnn.fnn_fc1.weight
        print(f"FCNN weights: {fnn_weights}")
        if self.controlled_system:
            gnn_weights = self.sdnn.gnn_fc1.weight
            print(f"GCNN weights: {gnn_weights}")    



        # plot resulting loss
        plt.plot(self.loss_batches)
        plt.ylabel('loss')
        plt.xlabel('nb. batches')
        plt.title(f"Learning rate: {self.learning_rate}, nb. epochs: {self.nb_epochs}")
        plt.show()


def learnSystemDynamics():
    ld = LearnDynamics()
    ld.optimize()
    ld.plot_losses()

if __name__ == "__main__":
    learnSystemDynamics()      



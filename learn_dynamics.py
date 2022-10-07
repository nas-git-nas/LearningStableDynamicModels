import numpy as np
import time
from datetime import datetime
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from damped_harmonic_oscillator import DampedHarmonicOscillator
from csdnn import CSDNN

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"
device = torch.device(dev)  

#torch.autograd.set_detect_anomaly(True)


class LearnDynamics():
    def __init__(self):
        # save model
        t = datetime.now()
        self.model_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
        self.model_dir = os.path.join("models", self.model_name)
        os.mkdir(self.model_dir)
        
        # neural network parameters
        self.learning_rate = 0.01
        self.nb_epochs = 200
        self.nb_batches = 300
        self.batch_size = 1000
        

        # dynamic model parameters
        self.alpha = 0.1 # constant ???
        self.loss_constant = 1
        self.controlled_system = True
        self.lyapunov_correction = True

        # real dynamic system
        self.dho = DampedHarmonicOscillator(dev=device)

        # modelled dynamic system
        self.sdnn = CSDNN(controlled_system=self.controlled_system, lyapunov_correction=self.lyapunov_correction, dev=device)
        self.sdnn.to(device)
        self.optimizer = torch.optim.SGD(self.sdnn.parameters(), lr=self.learning_rate)
        self.loss_batches = np.zeros((self.nb_epochs*self.nb_batches))
        self.loss_epochs = np.zeros((self.nb_epochs))

    def optimize(self):
        """
        Description: optimize nb_epochs times the model with all data (batch_size*nb_batches)
        """
        # generate data set
        self.dho.generate_data(self.batch_size, self.nb_batches)

        start_time = time.time()
        for j in range(self.nb_epochs):
            # get all of the data
            X, U, dX_real = self.dho.getData()

            for i in range(self.nb_batches):
                # forward pass through models
                dX_X = self.sdnn.forward(X[i,:,:], U[i,:,:]) # output of FCNN if input X (n)

                # calc. loss
                loss = self.loss_function(dX_X, dX_real[i,:,:])
                self.loss_batches[self.nb_batches*j+i] = loss.item()

                # backwards pass through models
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if j == 50:
                self.learning_rate = 0.005

            if j == 100:
                self.learning_rate = 0.0001

            if j == 150:
                self.learning_rate = 0.00005

            self.loss_epochs[j] = np.mean(self.loss_batches[self.nb_batches*j:self.nb_batches*(j+1)])
            print(f"Epoch {j}: Avg. loss = {self.loss_epochs[j]}, lr = {self.learning_rate}")
            
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

    def saveModel(self):
        # save model parameters
        torch.save(self.sdnn.state_dict(), os.path.join(self.model_dir, self.model_name+"_model"))

        # save training parameters
        with open(os.path.join(self.model_dir, self.model_name+"_log"), 'w') as f:
            f.write("--name:\n" + self.model_name + "\n\n")
            f.write("--notes:\n\n\n")
            f.write("--controlled system:\n" + str(self.controlled_system) + "\n\n")
            f.write("--lyapunov correction:\n" + str(self.lyapunov_correction) + "\n\n")
            f.write("--learning rate:\n" + str(self.learning_rate) + "\n\n")
            f.write("--number of epoches:\n" + str(self.nb_epochs) + "\n\n")
            f.write("--number of batches:\n" + str(self.nb_batches) + "\n\n")
            f.write("--number of samples per batch:\n" + str(self.batch_size) + "\n\n")
            f.write("--Losses:\n")
            for loss in self.loss_batches:
                f.write(str(loss) + "\n")

    # TODO: create fct to load a model
    # def loadModel(self, name):


    def printResults(self):
        """
        Description: plot losses and print some weights"""

        # create new test data set
        self.dho.generate_data(self.batch_size, nb_batches=1)

        # test model on test data set
        test_X, test_U, test_dX_real = self.dho.getData()
        test_dX_X = self.sdnn.forward(test_X[0,:,:], test_U[0,:,:])
        loss = self.loss_function(test_dX_X, test_dX_real[0,:,:])
        print(f"Error on testing set = {loss}")

        fnn_weights = self.sdnn.fnn_fc1.weight
        print(f"FCNN weights: {fnn_weights}")
        if self.controlled_system:
            gnn_weights = self.sdnn.gnn_fc1.weight
            print(f"GCNN weights: {gnn_weights}")    

    def plotResults(self):

        plot_x_min = self.dho.x_min - (self.dho.x_max-self.dho.x_min)/4
        plot_x_max = self.dho.x_max + (self.dho.x_max-self.dho.x_min)/4
        plot_dx_min = self.dho.dx_min - (self.dho.dx_max-self.dho.dx_min)/4
        plot_dx_max = self.dho.dx_max + (self.dho.dx_max-self.dho.dx_min)/4


        # define range of plot
        x_range = torch.arange(plot_x_min, plot_x_max+0.1, 0.1).to(device)
        dx_range = torch.arange(plot_dx_min, plot_dx_max+0.1, 0.1).to(device)

        # create equal distributed state vectors
        x_vector = x_range.tile((dx_range.size(0),))
        dx_vector = dx_range.repeat_interleave(x_range.size(0))
        X = torch.zeros(x_vector.size(0),self.dho.D).to(device)
        X[:,0] = x_vector
        X[:,1] = dx_vector

        # define control input
        U_zero = torch.zeros((X.shape[0],self.dho.M))
        U_max = torch.ones((X.shape[0],self.dho.M)) * self.dho.u_max

        fig, axs = plt.subplots(nrows=4, ncols=2, figsize =(12, 12))

        self.plotLoss(axs[0,0], axs[0,1])
        if self.lyapunov_correction:
            self.plotLyapunov(axs[1,0], axs[1,1], X, x_range, dx_range)
        self.plotDynamics(axs[2,0], axs[2,1], X, U_zero)
        if self.controlled_system:
            self.plotDynamics(axs[3,0], axs[3,1], X, U_max)

        plt.savefig(os.path.join(self.model_dir, self.model_name + "_figure"))

    def plotLoss(self, ax1, ax2):
        # plot resulting loss
        # axis_epochs = np.arange(self.nb_epochs*self.nb_batches)/self.nb_batches
        
        ax1.set_title(f"Learning rate: {self.learning_rate}, nb. epochs: {self.nb_epochs}")
        ax1.set_xlabel('nb. batches')
        ax1.set_ylabel('loss')
        ax1.plot(self.loss_epochs)

        ax2.set_title(f"Learning rate: {self.learning_rate}, nb. epochs: {self.nb_epochs}")
        ax2.set_yscale("log")
        ax2.set_xlabel('nb. batches')
        ax2.set_ylabel('log(loss)')
        ax2.plot(self.loss_epochs)

    def plotLyapunov(self, ax1, ax2, X, x_range, dx_range):
        # TODO: plot norm2 of lyapunov ||V||2

        # calc. lyapunov fct. of each state
        with torch.no_grad():
            V = self.sdnn.forwardLyapunov(X)

        X_contour, Y_contour = np.meshgrid(x_range, dx_range)
        Z_contour = np.array(V.reshape(dx_range.size(0),x_range.size(0)))
        
        ax1.set_title('Lyapunov fct. (V)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('dx')
        contours = ax1.contour(X_contour, Y_contour, Z_contour)
        ax1.clabel(contours, inline=1, fontsize=10)
        ax1.set_aspect('equal')

    def plotDynamics(self, ax1, ax2, X, U):

        # calc. learned system dynamics
        with torch.no_grad():
            dX_opt = self.sdnn.forward(X, U)      

        # calc. real system dynamics
        dX_real = self.dho.calc_dX_X(X.cpu(), U)

        ax1.set_title('Real dynamics (dX_real, U='+str(U[0,:].numpy())+')')
        ax1.set_xlabel('x')
        ax1.set_ylabel('dx')
        ax1.quiver(X[:,0], X[:,1], dX_real[:,0], dX_real[:,1], scale=None)
        ax1.set_aspect('equal')  
        rect_training = patches.Rectangle((self.dho.x_min,self.dho.dx_min), width=(self.dho.x_max-self.dho.x_min), \
                                            height=(self.dho.dx_max-self.dho.dx_min), facecolor='none', edgecolor="g")     
        ax1.add_patch(rect_training)

        ax2.set_title('Learned dynamics (dX_opt, U='+str(U[0,:].numpy())+')')
        ax2.set_xlabel('x')
        ax2.set_ylabel('dx')
        ax2.quiver(X[:,0], X[:,1], dX_opt[:,0], dX_opt[:,1], scale=None)
        ax2.set_aspect('equal')
        rect_training = patches.Rectangle((self.dho.x_min,self.dho.dx_min), width=(self.dho.x_max-self.dho.x_min), \
                                            height=(self.dho.dx_max-self.dho.dx_min), facecolor='none', edgecolor="g")
        ax2.add_patch(rect_training)


def learnSystemDynamics():
    ld = LearnDynamics()
    ld.optimize()
    ld.printResults()
    ld.saveModel()
    ld.plotResults()

    

if __name__ == "__main__":
    learnSystemDynamics()      



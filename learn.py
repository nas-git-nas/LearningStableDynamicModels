import numpy as np
import time
from datetime import datetime
import os
import shutil
import torch




# torch.autograd.set_detect_anomaly(True)


class Learn():
    def __init__(self, system, model, dev, model_type):
        # save model
        self.model_type = model_type
        t = datetime.now()
        self.model_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
        self.model_dir = os.path.join("models", self.model_type, self.model_name)
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.mkdir(self.model_dir)
        
        # learning parameters
        self.learning_rate = 0.01
        self.nb_epochs = 10
        self.nb_batches = 80
        self.batch_size = 512 
        self.testing_share = 0.1 # used to split data in training and testing sets
        self.device = dev
                                      
        self.sys = system
        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # logging data
        self.losses_tr = []
        self.losses_te = []



    def optimize(self):
        """
        Description: optimize nb_epochs times the model with all data (batch_size*nb_batches)
        """
        # load/generate data set
        if self.model_type=="DHO" or self.model_type=="CSTR":
            self.sys.generateData(self.batch_size*self.nb_batches)
        elif self.model_type=="Holohover":
            self.sys.loadData()

        
        for j in range(self.nb_epochs):
            start_time = time.time()

            # get data and split it into training and testing sets
            X_tr, X_te, U_tr, U_te, dX_tr, dX_te = self.splitData()
            print(X_tr.shape, U_tr.shape, dX_tr.shape)
            print(X_te.shape, U_te.shape, dX_te.shape)

            loss_tr = 0
            for X, U, dX_real in self.iterData(X_tr, U_tr, dX_tr):
                print(X.shape, U.shape, dX_real.shape)
                
                # forward pass through models
                dX_X = self.model.forward(X, U) # output of FCNN if input X (n)               

                # calc. loss
                loss = self.loss_function(dX_X, dX_real)
                loss_tr += loss

                # backwards pass through models
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # avg. training losses
            self.losses_tr.append(loss / np.ceil(X_tr.shape[0]/self.batch_size))

            # evaluate testing set
            dX_X = self.model.forward(X_te, U_te)
            self.losses_te.append(self.loss_function(dX_X, dX_te))

            # print results
            end_time =time.time()
            print(f"Epoch {j}: \telapse time = {np.round(end_time-start_time,3)}, \ttesting loss = {self.losses_te[-1]}, \ttraining loss = {self.losses_tr[-1]}")

    def loss_function(self, dX_X, dX_real):
        """
        Description: calc. average loss of batch X
        In dX_X: approx. of system dynamics (b x n)
        In dX_real: real system dynamics (b x n)
        Out L: average loss of batch X (scalar)
        """
        return (1/dX_X.shape[0]) * torch.sum(torch.square(dX_X-dX_real))

    def splitData(self):
        """
        Get data from system class and split into training and testing sets
        Returns:
            X_tr: state training set, tensor (N*(1-testing_share),D)
            X_te: state testing set, tensor (N*testing_share,D)
            U_tr: control input training set, tensor (N*(1-testing_share),M)
            U_te: control input testing set, tensor (N*testing_share,M)
            dX_tr: dynmaics training set, tensor (N*(1-testing_share),D)
            dX_te: dynmaics testing set, tensor (N*testing_share,D)
        """
        X, U, dX = self.sys.getData(u_map=True)

        # randomize order
        rand_order = torch.randperm(X.shape[0])
        X = X[rand_order,:]
        U = U[rand_order,:]
        dX = dX[rand_order,:]

        # split into training and testing sets
        split_idx = int((1-self.testing_share)*X.shape[0])
        X_tr = X[:split_idx,:]
        X_te = X[split_idx:,:]
        U_tr = U[:split_idx,:]
        U_te = U[split_idx:,:]
        dX_tr = dX[:split_idx,:]
        dX_te = dX[split_idx:,:]

        return X_tr, X_te, U_tr, U_te, dX_tr, dX_te

    def iterData(self, X, U, dX):
        """
        Iterate over data set, batch_size of last batch may vary
        Args:
            X: state, tensor (N,D)
            U: control input, tensor (N,M)
            dX: dynamics, tensor (N,D)
        Yields:
            X: state, tensor (batch_size,D)
            U: control input, tensor (batch_size,M)
            dX: dynamics, tensor (batch_size,D)
            """

        # calc. number of batches of given data set and given batch size
        nb_batches = int(np.ceil(X.shape[0]/self.batch_size))

        # yield one batch after each other
        for b in range(nb_batches-1):
            b_range = torch.arange(b*self.batch_size, (b+1)*self.batch_size)
            yield X[b_range,:], U[b_range,:], dX[b_range,:]

        # yield last batch that may be smaller than batch_size
        b_range = torch.arange((nb_batches-1)*self.batch_size, X.shape[0])
        yield X[b_range,:], U[b_range,:], dX[b_range,:]

    def testModel(self):
        # create new test data set
        self.sys.generateData(self.batch_size, nb_batches=1)

        # test model on test data set
        test_X, test_U, test_dX_real = self.sys.getData()
        test_dX_X = self.model.forward(test_X[0,:,:], test_U[0,:,:])
        return self.loss_function(test_dX_X, test_dX_real[0,:,:])        

    def saveModel(self):
        # save model parameters
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.model_name+"_model"))

        # save training parameters
        with open(os.path.join(self.model_dir, self.model_name+"_log"), 'w') as f:
            f.write("--name:\n" + self.model_name + "\n\n")
            f.write("--model type:\n" + self.model_type + "\n\n")
            f.write("--controlled system:\n" + str(self.model.controlled_system) + "\n\n")
            f.write("--lyapunov correction:\n" + str(self.model.lyapunov_correction) + "\n\n")
            f.write("--learning rate:\n" + str(self.learning_rate) + "\n\n")
            f.write("--number of epoches:\n" + str(self.nb_epochs) + "\n\n")
            f.write("--number of batches:\n" + str(self.nb_batches) + "\n\n")
            f.write("--number of samples per batch:\n" + str(self.batch_size) + "\n\n")
            f.write("--FNN fc1 weights:\n" + str(self.model.fnn_fc1.weight) + "\n\n")
            f.write("--GNN fc1 weights:\n" + str(self.model.gnn_fc1.weight) + "\n\n")
            f.write("--GNN fc1 bias:\n" + str(self.model.gnn_fc1.bias) + "\n\n")
            f.write("--Testing loss:\n")
            for loss in self.losses_te:
                f.write(str(loss) + "\n")
            f.write("--Training losses:\n")
            for loss in self.losses_tr:
                f.write(str(loss) + "\n")

    def printResults(self):
        """
        Description: plot losses and print some weights"""
        loss = self.testModel()
        print(f"Error on testing set = {loss}")

        fnn_weights = self.model.fnn_fc1.weight
        print(f"FCNN weights: {fnn_weights}")
        if self.controlled_system:
            gnn_weights = self.model.gnn_fc1.weight
            print(f"GCNN weights: {gnn_weights}")    

    
    



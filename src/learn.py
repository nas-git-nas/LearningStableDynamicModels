import numpy as np
import time
from datetime import datetime
import os
import shutil
import torch
import matplotlib.pyplot as plt


# torch.autograd.set_detect_anomaly(True)


class Learn():
    def __init__(self, args, dev, system, model):
        self.args = args

        # data
        self.load_data = args.load_data

        # save model
        self.model_type = args.model_type
        t = datetime.now()
        self.model_name = t.strftime("%Y%m%d") + "_" + t.strftime("%H%M")
        self.model_dir = os.path.join("models", self.model_type, self.model_name)
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.mkdir(self.model_dir)
        
        # learning parameters
        self.learning_rate = args.learning_rate
        self.nb_epochs = args.nb_epochs
        self.nb_batches = args.nb_batches
        self.batch_size = args.batch_size
        self.testing_share = args.testing_share
        self.device = dev
        self.lossCoeff = np.zeros((system.M))
                                      
        self.sys = system
        self.model = model
        self.model.to(self.device)

        # choose a different learning rates for params
        # model_params = self.model.parameters()
        model_params = [ { 'params':self.model.tnn_sig2thr_fcts[i].weight, 'lr':1e-4 } 
                            for i in range(len(self.model.tnn_sig2thr_fcts))]
        model_params.append( {'params': self.model.center_of_mass, 'lr':1e-6 } )
        model_params.append( {'params': self.model.inertia, 'lr': 1e-7 } )
        # self.optimizer = torch.optim.SGD(model_params, lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(model_params, lr=self.learning_rate)

        # logging data
        self.losses_tr = []
        self.losses_te = []
        self.abs_error_te = []



    def optimize(self):
        """
        Description: optimize nb_epochs times the model with all data (batch_size*nb_batches)
        """
        # load/generate data set
        if self.load_data:
            self.sys.loadData()            
        else:
            self.sys.generateData(self.batch_size*self.nb_batches)


        # get data and split it into training and testing sets
        X_data, U_data, dX_data = self.sys.getData()
        self.calcLossCoeff(dX_data)
        # print(f"shape X_data: {X_data.shape}, U_data: {U_data.shape}, dX_data: {dX_data.shape}")
        X_tr, X_te, U_tr, U_te, dX_tr, dX_te = self.splitData(X_data, U_data, dX_data)

        # evaluate testing set before training
        self.evaluate(X_te=X_te, dX_te=dX_te, U_te=U_te)
        
        for j in range(self.nb_epochs):
            start_time = time.time()

            loss_tr = []
            for X, U, dX_real in self.iterData(X_tr, U_tr, dX_tr):
              
                # forward pass through models
                dX_X = self.model.forward(X, U) # (N,D)
                
                # print(f"U min: {torch.min(U)}, U max: {torch.max(U)}")
                # print(f"shape X: {X.shape}, U: {U.shape}, dX: {dX_X.shape}, dX_real: {dX_real.shape}")          

                # calc. loss
                loss = self.lossFunction(dX_X, dX_real)
                loss_tr.append(loss.detach().clone().float())

                # print(f"batch loss: {loss.detach().clone().float()}") 
                # print(f"mean error: {(torch.mean(torch.square(dX_X-dX_real), axis=0) / self.lossCoeff).detach().numpy()}")

                # plt.plot(dX_X[:,4].detach().numpy(), label="ddx")
                # plt.plot(dX_real[:,4].detach().numpy(), label="ddx real")
                # plt.legend()
                # plt.show()

                # backwards pass through models
                if self.args.learn_center_of_mass or self.args.learn_inertia or self.args.learn_signal2thrust:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # print(f"trainting loss = {loss}")

            # avg. training losses
            self.losses_tr.append(np.mean(loss_tr))

            # evaluate testing set
            self.evaluate(X_te=X_te, dX_te=dX_te, U_te=U_te)

            # print results
            end_time =time.time()
            print(f"Epoch {j+1}: \telapse time = {np.round(end_time-start_time,3)}, \ttesting loss = {self.losses_te[-1]}, \ttraining loss = {self.losses_tr[-1]}")

        print(f"Center of mass: {self.model.center_of_mass}, Center of mass init.: {self.model.init_center_of_mass}\n")
        print(f"Inertia: {np.round(self.model.inertia[0].detach().numpy(),6)}, Inertia init.: {np.round(self.model.init_inertia[0].detach().numpy(),6)}\n")
        print(f"Signal2thrust weights: {self.model.tnn_sig2thr_fcts[0].weight}, init. {self.model.init_signal2thrust[0]}\n")

    def calcLossCoeff(self, dX):
        self.lossCoeff = torch.std(torch.pow(dX, 2), axis=0)

    def lossFunction(self, dX_X, dX_real):
        """
        Description: calc. average loss of batch X
        In dX_X: approx. of system dynamics (b x n)
        In dX_real: real system dynamics (b x n)
        Out L: average loss of batch X (scalar)
        """
        mse = torch.sum( torch.mean(torch.square(dX_X-dX_real), axis=0) / self.lossCoeff )
        # print(f"com: {self.model.center_of_mass}")

        if self.args.regularize_center_of_mass:
            mse += torch.linalg.norm(self.model.center_of_mass-self.model.init_center_of_mass)**2

        if self.args.regularize_inertia:
            # print(f"inertia: {self.model.inertia[0]}")
            mse += 0.001*torch.abs(self.model.inertia[0]-self.model.init_inertia[0]) / self.model.init_inertia[0]


        return mse

    def evaluate(self, X_te, dX_te, U_te):
        """
        Evaluation step: calc. loss and abs. error on testing set
        Args:
            X_te: testing set state, tensor (N_te,D)
            dX_te: testing set dynamics, tensor (N_te,D)
            U_te: testing set control input, tensor (N_te,M)
        """
        with torch.no_grad():
            dX_X = self.model.forward(X_te, U_te)

        self.losses_te.append(self.lossFunction(dX_X, dX_te).detach().clone().float())
        self.abs_error_te.append(torch.mean(torch.abs(dX_X - dX_te), axis=0).detach().numpy())

    def splitData(self, X, U, dX):
        """
        Get data from system class and split into training and testing sets
        Args:
            X: state, tensor (N,D)
            U: control input, tensor (N,M)
            dX: dynmaics, tensor (N,D)
        Returns:
            X_tr: state training set, tensor (N*(1-testing_share),D)
            X_te: state testing set, tensor (N*testing_share,D)
            U_tr: control input training set, tensor (N*(1-testing_share),M)
            U_te: control input testing set, tensor (N*testing_share,M)
            dX_tr: dynmaics training set, tensor (N*(1-testing_share),D)
            dX_te: dynmaics testing set, tensor (N*testing_share,D)
        """
        X = X.clone().detach()
        U = U.clone().detach()
        dX = dX.clone().detach()

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
        X = X.clone().detach()
        U = U.clone().detach()
        dX = dX.clone().detach()

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
        return self.lossFunction(test_dX_X, test_dX_real[0,:,:])        

    def saveModel(self):
        # save model parameters
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.model_name+"_model"))

        # save training parameters
        with open(os.path.join(self.model_dir, self.model_name+"_log"), 'w') as f:
            f.write("--name:\n" + self.model_name + "\n\n")
            f.write("--model type:\n" + self.model_type + "\n\n")
            if self.args.black_model:
                f.write("--controlled system:\n" + str(self.args.controlled_system) + "\n\n")
                f.write("--lyapunov correction:\n" + str(self.args.lyapunov_correction) + "\n\n")
            f.write("--learning rate:\n" + str(self.learning_rate) + "\n\n")
            f.write("--number of epoches:\n" + str(self.nb_epochs) + "\n\n")
            f.write("--number of batches:\n" + str(self.nb_batches) + "\n\n")
            f.write("--number of samples per batch:\n" + str(self.batch_size) + "\n\n")
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

    
    



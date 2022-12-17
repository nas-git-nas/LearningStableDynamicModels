import numpy as np
import time
from abc import abstractmethod
from datetime import datetime
import os
import shutil
import torch
import matplotlib.pyplot as plt


# torch.autograd.set_detect_anomaly(True)


class Learn():
    def __init__(self, args, dev, system, model):
        self.args = args
        self.device = dev                                    
        self.sys = system
        self.model = model.to(self.device)

        # loss params
        self.lossCoeff = []

        # logging data
        self.losses_tr = []
        self.losses_te = []
        self.abs_error_te = []

    @abstractmethod
    def forward(self, X, U):
        raise ValueError(f"Function not implemented")

    @abstractmethod
    def lossFunction(self, dX_X, dX_real):
        raise ValueError(f"Function not implemented")

    @abstractmethod
    def evaluate(self, X_te, dX_te, U_te):
        raise ValueError(f"Function not implemented")

    def getData(self):
        if self.args.load_data:
            self.sys.loadData()            
        else:
            self.sys.generateData(self.batch_size*self.nb_batches)

        X_data, U_data, dX_data = self.sys.getData()
        self.lossCoeff = torch.std(torch.pow(X_data, 2), axis=0)
        X_tr, X_te, U_tr, U_te, dX_tr, dX_te = self.splitData(X_data, U_data, dX_data)

        return X_tr, X_te, U_tr, U_te, dX_tr, dX_te 

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
        split_idx = int((1-self.args.testing_share)*X.shape[0])
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
        nb_batches = int(np.ceil(X.shape[0]/self.args.batch_size))

        # yield one batch after each other
        for b in range(nb_batches-1):
            b_range = torch.arange(b*self.args.batch_size, (b+1)*self.args.batch_size)
            yield X[b_range,:], U[b_range,:], dX[b_range,:]

        # yield last batch that may be smaller than batch_size
        b_range = torch.arange((nb_batches-1)*self.args.batch_size, X.shape[0])
        yield X[b_range,:], U[b_range,:], dX[b_range,:]

    def optimize(self):
        X_tr, X_te, U_tr, U_te, dX_tr, dX_te = self.getData()

        self.evaluate(X_te=X_te, dX_te=dX_te, U_te=U_te)

        self.losses_tr = []
        self.losses_te = []
        self.abs_error_te = []
        for j in range(self.args.nb_epochs):
            start_time = time.time()

            loss_tr = []
            for X, U, dX_real in self.iterData(X_tr, U_tr, dX_tr):
              
                dX_X = self.forward(X, U) # (N,D) 

                loss = self.lossFunction(dX_X, dX_real)
                loss_tr.append(loss.detach().clone().float())

                # loss_v = torch.mean(torch.square(dX_X-dX_real) , axis=0) / self.lossCoeff
                # print(f"training loss: {loss}, verification: {loss_v}, loss coeff.: {self.lossCoeff}")
                # tr_abs_error = torch.mean(torch.abs(dX_X - dX_real), axis=0)
                # print(f"training error: {tr_abs_error}")
                # plt.plot(dX_X[:,3].detach().numpy(), label="est") 
                # plt.plot(dX_real[:,3].detach().numpy(), label="real") 
                # plt.show()               

                # if self.args.learn_center_of_mass or self.args.learn_inertia or self.args.learn_signal2thrust or args.learn_mass:                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.losses_tr.append(np.mean(loss_tr))
            self.evaluate(X_te=X_te, dX_te=dX_te, U_te=U_te)
            print(f"Epoch {j+1}: \telapse time = {np.round(time.time()-start_time,3)}, \ttesting loss = {self.losses_te[-1]}, \
                    \ttraining loss = {self.losses_tr[-1]}")

    def saveModel(self):
        # save model parameters
        torch.save(self.model.state_dict(), os.path.join(self.args.dir_path, self.args.model_type+"_model"))


class LearnGreyModel(Learn):
    def __init__(self, args, dev, system, model):
        Learn.__init__(self, args=args, dev=dev, system=system, model=model)

        model_params = [ { 'params':self.model.sig2thr_fcts[i].weight, 'lr':1e-4 } 
                            for i in range(len(self.model.sig2thr_fcts))]
        model_params.append( {'params': self.model.center_of_mass, 'lr':1e-6 } )
        model_params.append( {'params': self.model.mass, 'lr':1e-5 } )
        model_params.append( {'params': self.model.inertia, 'lr': 1e-7 } )
        model_params.append( {'params': self.model.motors_vec, 'lr': 1e-5 } )
        model_params.append( {'params': self.model.motors_pos, 'lr': 1e-5 } )
        self.optimizer = torch.optim.Adam(model_params, lr=self.args.learning_rate)

    def forward(self, X, U):
        dX_X = self.model.forward(X, U)
        return dX_X
        
        print(f"Center of mass: {self.model.center_of_mass}, Center of mass init.: {self.model.init_center_of_mass}\n")
        print(f"Mass: {np.round(self.model.mass[0].detach().numpy(),8)}, Mass init.: {np.round(self.model.init_mass[0].detach().numpy(),8)}\n")
        print(f"Inertia: {np.round(self.model.inertia[0].detach().numpy(),6)}, Inertia init.: {np.round(self.model.init_inertia[0].detach().numpy(),6)}\n")
        print(f"Signal2thrust weights: {self.model.tnn_sig2thr_fcts[0].weight}, init. {self.model.init_signal2thrust[0]}\n")
        print(f"Motor vec: {self.model.motors_vec}, \nMotor vec. init. \n{self.model.init_motors_vec}\n")
        print(f"Motor pos: {self.model.motors_pos}, \nMotor pos. init. \n{self.model.init_motors_pos}\n")

    def lossFunction(self, dX_X, dX_real):
        """
        Description: calc. average loss of batch X
        In dX_X: approx. of system dynamics (b x n)
        In dX_real: real system dynamics (b x n)
        Out L: average loss of batch X (scalar)
        """
        mse = torch.sum( torch.mean(torch.square(dX_X-dX_real), axis=0) / self.lossCoeff )
        # print(f"com: {self.model.center_of_mass}")

        # if self.args.regularize_center_of_mass:
        #     mse += torch.linalg.norm(self.model.center_of_mass-self.model.init_center_of_mass)**2

        # if self.args.regularize_inertia:
        #     # print(f"inertia: {self.model.inertia[0]}")
        #     mse += 0.001*torch.abs(self.model.inertia[0]-self.model.init_inertia[0]) / self.model.init_inertia[0]


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

class LearnCorrection(Learn):
    def __init__(self, args, dev, system, model, base_model):
        Learn.__init__(self, args=args, dev=dev, system=system, model=model)

        self.base_model = base_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def forward(self, X, U):
        with torch.no_grad():
            dX_pred = self.base_model.forward(X, U)     

        acc_cor = self.model.forward(X=X, U=U)
        dX_X = torch.concat([torch.zeros(acc_cor.shape), acc_cor], dim=1)
        dX_X = dX_X + dX_pred.detach().clone() 
        return dX_X

    def lossFunction(self, dX_X, dX_real):
        """
        Description: calc. average loss of batch X
        In dX_X: approx. of system dynamics (b x n)
        In dX_real: real system dynamics (b x n)
        Out L: average loss of batch X (scalar)
        """
        mse = torch.sum( torch.mean(torch.square(dX_X-dX_real), axis=0) / self.lossCoeff )
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
            dX_pred = self.base_model.forward(X_te, U_te)
            acc_cor = self.model.forward(X=X_te, U=U_te)
            dX_X = torch.concat([torch.zeros(acc_cor.shape), acc_cor], dim=1)
            dX_X = dX_X + dX_pred.detach().clone()
        self.losses_te.append(self.lossFunction(dX_X, dX_te))
        self.abs_error_te.append(torch.mean(torch.abs(dX_X - dX_te), axis=0).detach().numpy())    


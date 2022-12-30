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
        self.metrics = { "losses_tr":[], "losses_te":[], "abs_error":[], "rms_error":[], "std_error":[] }

    @abstractmethod
    def forward(self, X, U):
        raise ValueError(f"Function not implemented")

    @abstractmethod
    def lossFunction(self, dX_X, dX_real):
        raise ValueError(f"Function not implemented")

    @abstractmethod
    def evaluate(self, X_te, dX_te, U_te):
        raise ValueError(f"Function not implemented")

    @abstractmethod
    def printMetrics(self, epoch, elapse_time):
        raise ValueError(f"Function not implemented")

    def getData(self):
        if self.args.load_data:
            self.sys.loadData()            
        else:
            self.sys.generateData(self.args.batch_size*self.args.nb_batches)

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

        self.metrics = { "losses_tr":[], "losses_te":[], "abs_error":[], "rms_error":[], "std_error":[] }
        self.evaluate(X_te=X_te, dX_te=dX_te, U_te=U_te)
        self.metrics["losses_tr"].append(self.metrics["losses_te"][0])
        self.printMetrics(epoch=0, elapse_time=0.0)

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

            self.metrics["losses_tr"].append(np.mean(loss_tr))
            self.evaluate(X_te=X_te, dX_te=dX_te, U_te=U_te)
            self.printMetrics(epoch=j+1, elapse_time=time.time()-start_time)

        # print(f"motors_vec norm error: {torch.abs(torch.linalg.norm(self.model.motors_vec, dim=1)-torch.ones(self.sys.M)).detach().numpy()}")
        # print(f"motors_pos error: {torch.linalg.norm(self.model.motors_pos-self.model.init_motors_pos, dim=1)}")

    def saveModel(self):
        # save model parameters
        torch.save(self.model.state_dict(), os.path.join(self.args.dir_path, self.args.model_type+"_model"))

class LearnStableModel(Learn):
    def __init__(self, args, dev, system, model):
        Learn.__init__(self, args=args, dev=dev, system=system, model=model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def forward(self, X, U):
        dX_X = self.model.forward(X, U)
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
        X_te, dX_te, U_te = X_te.detach().clone(), dX_te.detach().clone(), U_te.detach().clone()
        with torch.no_grad():
            dX_X = self.model.forward(X_te, U_te)

        self.metrics["losses_te"].append(self.lossFunction(dX_X, dX_te).detach().clone().float())
        self.metrics["abs_error"].append(torch.mean(torch.abs(dX_X - dX_te), axis=0).detach().numpy())
        self.metrics["rms_error"].append(torch.sqrt(torch.mean(torch.pow(dX_X - dX_te, 2), axis=0)).detach().numpy())
        self.metrics["std_error"].append(torch.std(dX_X - dX_te, axis=0).detach().numpy())

    def printMetrics(self, epoch, elapse_time):
        loss_te = self.metrics["losses_te"][-1]
        loss_tr = self.metrics["losses_tr"][-1]
        abs_error = self.metrics["abs_error"][-1]
        rms_error = self.metrics["rms_error"][-1]
        std_error = self.metrics["std_error"][-1]
        print(f"Epoch {epoch}: \telapse time = {np.round(elapse_time,3)}, \ttesting loss = {loss_te}, \ttraining loss = {loss_tr}")
        print(f"Epoch {epoch}: \tabs error = {np.round(abs_error,4)}, \trms error = {np.round(rms_error,4)}, \tstd error = {np.round(std_error,4)}")


class LearnGreyModel(Learn):
    def __init__(self, args, dev, system, model):
        Learn.__init__(self, args=args, dev=dev, system=system, model=model)

        model_params = [ { 'params':self.model.sig2thr_fcts[i].weight, 'lr':args.lr_signal2thrust } 
                            for i in range(len(self.model.sig2thr_fcts))]
        model_params.append( {'params': self.model.center_of_mass, 'lr':args.lr_center_of_mass } )
        model_params.append( {'params': self.model.mass, 'lr':args.lr_mass } )
        model_params.append( {'params': self.model.inertia, 'lr':args.lr_inertia } )
        model_params.append( {'params': self.model.motors_vec, 'lr':args.lr_motors_vec } )
        model_params.append( {'params': self.model.motors_pos, 'lr':args.lr_motors_pos } )
        self.optimizer = torch.optim.Adam(model_params, lr=self.args.learning_rate)

    def forward(self, X, U):
        dX_X = self.model.forward(X, U)
        return dX_X

    def lossFunction(self, dX_X, dX_real):
        """
        Description: calc. average loss of batch X
        In dX_X: approx. of system dynamics (b x n)
        In dX_real: real system dynamics (b x n)
        Out L: average loss of batch X (scalar)
        """
        mse = torch.sum( torch.mean(torch.square(dX_X-dX_real), axis=0) / self.lossCoeff )
        mse += 10 * torch.linalg.norm(self.model.motors_pos-self.model.init_motors_pos)
        mse += 0.1 * torch.sum( torch.abs(torch.linalg.norm(self.model.motors_vec, dim=1)-torch.ones(self.sys.M)))

        return mse

    def evaluate(self, X_te, dX_te, U_te):
        """
        Evaluation step: calc. loss and abs. error on testing set
        Args:
            X_te: testing set state, tensor (N_te,D)
            dX_te: testing set dynamics, tensor (N_te,D)
            U_te: testing set control input, tensor (N_te,M)
        """
        X_te, dX_te, U_te = X_te.detach().clone(), dX_te.detach().clone(), U_te.detach().clone()
        with torch.no_grad():
            dX_X = self.model.forward(X_te, U_te)

        self.metrics["losses_te"].append(self.lossFunction(dX_X, dX_te).detach().clone().float())
        self.metrics["abs_error"].append(torch.mean(torch.abs(dX_X - dX_te), axis=0).detach().numpy())
        self.metrics["rms_error"].append(torch.sqrt(torch.mean(torch.pow(dX_X - dX_te, 2), axis=0)).detach().numpy())
        self.metrics["std_error"].append(torch.std(dX_X - dX_te, axis=0).detach().numpy())

    def printMetrics(self, epoch, elapse_time):
        loss_te = self.metrics["losses_te"][-1]
        loss_tr = self.metrics["losses_tr"][-1]
        abs_error = self.metrics["abs_error"][-1][3:6]
        rms_error = self.metrics["rms_error"][-1][3:6]
        std_error = self.metrics["std_error"][-1][3:6]
        print(f"Epoch {epoch}: \telapse time = {np.round(elapse_time,3)}, \ttesting loss = {loss_te}, \ttraining loss = {loss_tr}")
        print(f"Epoch {epoch}: \tabs error = {np.round(abs_error,4)}, \trms error = {np.round(rms_error,4)}, \tstd error = {np.round(std_error,4)}")

class LearnCorrection(Learn):
    def __init__(self, args, dev, system, model, base_model):
        Learn.__init__(self, args=args, dev=dev, system=system, model=model)

        self.base_model = base_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr_cor)

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
        X_te, dX_te, U_te = X_te.detach().clone(), dX_te.detach().clone(), U_te.detach().clone()
        with torch.no_grad():
            dX_pred = self.base_model.forward(X_te, U_te)
            acc_cor = self.model.forward(X=X_te, U=U_te)
            dX_X = torch.concat([torch.zeros(acc_cor.shape), acc_cor], dim=1)
            dX_X = dX_X + dX_pred.detach().clone()
        self.metrics["losses_te"].append(self.lossFunction(dX_X, dX_te).detach().clone().float())
        self.metrics["abs_error"].append(torch.mean(torch.abs(dX_X - dX_te), axis=0).detach().numpy()) 
        self.metrics["rms_error"].append(torch.sqrt(torch.mean(torch.pow(dX_X - dX_te, 2), axis=0)).detach().numpy())
        self.metrics["std_error"].append(torch.std(dX_X - dX_te, axis=0).detach().numpy())   


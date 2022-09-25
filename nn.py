import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ICNN(nn.Module):
    def __init__(self):
        super(ICNN, self).__init__()

        # FCNN: model parameters
        fcnn_input_size = 2
        fcnn_output_size = 2

        # ICNN: model parameters
        icnn_input_size = 2
        icnn_hidden1_size = 60
        icnn_hidden2_size = 60
        icnnn_output_size = 1

        # FCNN: layers
        self.fcnn_fc1 = nn.Linear(fcnn_input_size, fcnn_output_size, bias=False)

        # ICNN: fully connected layers
        self.icnn_fc1 = nn.Linear(icnn_input_size, icnn_hidden1_size)
        self.icnn_fc2 = nn.Linear(icnn_hidden1_size, icnn_hidden2_size)
        self.icnn_fc3 = nn.Linear(icnn_hidden2_size, icnnn_output_size)

        # ICNN: input mapping
        self.icnn_im2 = nn.Linear(icnn_input_size, icnn_hidden2_size, bias=False)
        self.icnn_im3 = nn.Linear(icnn_input_size, icnnn_output_size, bias=False)
 
        # activation fcts.
        self.sp = nn.Softplus() #softplus (smooth approx. of ReLU)

        self.epsilon = 0.01

    def forward(self, X):
        # FCNN
        X_approx = self.fcnn_fc1(X)

        # ICNN
        x_icnn_fc1 = self.fc1(X)
        x_icnn_sp1 = self.sp(x_icnn_fc1)

        x_icnn_fc2 = self.icnn_fc2(x_icnn_sp1)
        x_icnn_im2 = self.icnn_im2(X)
        x_icnn_sp2 = self.sp(x_icnn_fc2 + x_icnn_im2)

        x_icnn_fc3 = self.fc3(x_icnn_sp2)
        x_icnn_im3 = self.im3(X)
        g_X = self.sp(x_icnn_fc3 + x_icnn_im3)
        
        return x_sp3

    def predict(self, x):
        logits = self.forward(x)
        return nn.Softmax(logits)

    def lyapunov(self, X, g_X, g_zero):
        """
        Description: calc. lyapunov fct. V
        In X: input batch (b x n)
        In g_X: output of ICNN if input X (b)
        In g_zero: output of ICNN if input x=0 (scalar)
        Out: lyapunov fct. V from batch (b)
        """
        return self.sigma_lyapunov(g_X, g_zero) + self.epsilon*torch.diagonal(X@X.T)

    def gradient_lyapunov(self, X, g_X):
        """
        Description: calc. gradient of lyapunov fct. V (b x n)
        In X: input batch (b x n)
        In g_X: output of ICNN if input X (b)
        Out dV: gradient of lyapunov fct. V (b x n)
        """
        # dV = np.zeros((X.shape))
        # x_zero = torch.zeros((X.shape[1]))
        # g_zero = self.forward(x_zero)
        # dsigma_lyap = self.derivative_sigma_lyapunov(g_X, g_zero)

        # for i in range(X.shape[0]): # loop through all input vectors in batch x
        #     J = torch.autograd.functional.jacobian(self.forward, X[i,:]) - torch.autograd.functional.jacobian(self.forward, x_zero)
            
        #     dV[i,:] = J@dsigma_lyap + 2*X[i,:]

        x_zero = torch.zeros((X.shape[1]))
        g_zero = self.forward(x_zero)
        J_X = torch.zeros(X.shape)
        J_zero = torch.zeros(X.shape)
        for i in range(X.shape[0]):
            J_X[i,:] = torch.autograd.functional.jacobian(self.forward, X[i,:])
            J_zero[i,:] = torch.autograd.functional.jacobian(self.forward, x_zero)
        dsigma_lyap = torch.tile(self.derivative_sigma_lyapunov(g_X, g_zero),(1,2))
        dV = dsigma_lyap*(J_X-J_zero) + 2*self.epsilon*X

        return dV

    def sigma_lyapunov(self, g_X, g_zero):
        """
        Description: enforces V(0)=0 (needed for positive definiteness of lyapunov ftc.)
        In g: output of ICNN with input x (b = batch size)
        In g_zero: output of ICNN with input x=0 (scalar)
        Out: lyapunov sigma (b)
        """
        sigma_lyap = torch.flatten(g_X)-g_zero[0]


        relu = nn.ReLU()
        sigma_lyap = relu(sigma_lyap)

        # for i, sig in enumerate(sigma_lyap):
        #     if sig <= 0:
        #         sigma_lyap[i] = 0
        #     elif sig < 1:
        #         sigma_lyap[i] = 0.5*sig*sig
        #     else:
        #         sigma_lyap[i] = sig - 0.5
        return sigma_lyap 

    def derivative_sigma_lyapunov(self, g, g_zero):
        """
        Description: enforces V(0)=0 (needed for positive definiteness of lyapunov ftc.)
        In g: output of ICNN with input x
        In g_zero: output of ICNN with input x=0
        Out: derivative of lyapunov sigma
        """
        dsigma_lyap = g-g_zero
        dsigma_lyap[dsigma_lyap<0] = 0
        dsigma_lyap[dsigma_lyap>=0] = 1        
        # dsigma_lyap[dsigma_lyap<=0] = 0
        # dsigma_lyap[dsigma_lyap>=1] = 1

        return dsigma_lyap  


def testICNN():
    # define parameters
    n_input = 4
    n_hidden1 = 60
    n_hidden2 = 60
    n_out = 1
    batch_size = 100
    learning_rate = 0.01
    n_epochs = 10000

    # create random data for testing
    data_x = torch.randn(batch_size, n_input)
    data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()

    # define model, loss and optimizer
    model = ICNN()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # run n_epochs optimizations
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred_y = model.forward(data_x)
        loss = loss_function(pred_y, data_y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # plot resulting loss
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()

if __name__ == "__main__":
    testICNN()
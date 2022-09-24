import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ICNN(nn.Module):
    def __init__(self):
        super(ICNN, self).__init__()

        # model parameters
        n_input = 4
        n_hidden1 = 60
        n_hidden2 = 60
        n_out = 1

        # fully connected layers
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_out)

        # input mapping
        self.im2 = nn.Linear(n_input, n_hidden2, bias=False)
        self.im3 = nn.Linear(n_input, n_out, bias=False)
 
        # softplus (smooth approx. of ReLU)
        self.sp = nn.Softplus()

        self.epsilon = 0.01

    def forward(self, x_in):
        # first layer
        x_fc1 = self.fc1(x_in)
        x_sp1 = self.sp(x_fc1)

        # second layer
        x_fc2 = self.fc2(x_sp1)
        x_im2 = self.im2(x_in)
        x_sp2 = self.sp(x_fc2 + x_im2)

        # third layer
        x_fc3 = self.fc3(x_sp2)
        x_im3 = self.im3(x_in)
        x_sp3 = self.sp(x_fc3 + x_im3)
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
        return self.sigma_lyapunov(g_X, g_zero) + self.epsilon*np.diagonal(X@X.T)

    def gradient_lyapunov(self, X, g_X):
        """
        Description: calc. gradient of lyapunov fct. V (b x n)
        In X: input batch (b x n)
        In g_X: output of ICNN if input X (b)
        Out dV: gradient of lyapunov fct. V (b x n)
        """
        dV = np.zeros((X.shape))
        x_zero = torch.zeros((X.shape[1]))
        g_zero = self.forward(x_zero)

        for i in range(x.shape[0]): # loop through all input vectors in batch x
            J = torch.autograd.functional.jacobian(self.forward, X[i,:]) - torch.autograd.functional.jacobian(self.forward, x_zero)
            dsigma_lyap = self.derivative_sigma_lyapunov(g_X, g_zero)
            dV[i,:] = J@dsigma_lyap + 2*X[i,:]

        return dV

    def sigma_lyapunov(self, g_X, g_zero):
        """
        Description: enforces V(0)=0 (needed for positive definiteness of lyapunov ftc.)
        In g: output of ICNN with input x (b = batch size)
        In g_zero: output of ICNN with input x=0 (scalar)
        Out: lyapunov sigma
        """
        sigma_lyap = g_X-g_zero
        for i, sig in enumerate(sigma_lyap):
            if sig <= 0:
                sigma_lyap[i] = 0
            elif sig < 1:
                sigma_lyap[i] = 0.5*sig*sig
            else:
                sigma_lyap[i] = sig - 0.5
        return sigma_lyap 

    def derivative_sigma_lyapunov(self, g, g_zero):
        """
        Description: enforces V(0)=0 (needed for positive definiteness of lyapunov ftc.)
        In g: output of ICNN with input x
        In g_zero: output of ICNN with input x=0
        Out: derivative of lyapunov sigma
        """
        dsigma_lyap = g-g_zero
        dsigma_lyap[dsigma_lyap<=0] = 0
        dsigma_lyap[dsigma_lyap>=1] = 1
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
    model = ICNN(n_input, n_hidden1, n_hidden2, n_out)
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
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SDNN(nn.Module):
    def __init__(self):
        super(SDNN, self).__init__()

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
        self.relu = nn.ReLU()
        self.sp = nn.Softplus() #softplus (smooth approx. of ReLU)

        self.epsilon = 0.01
        self.alpha = 0.1

    def forward(self, X):
        # FCNN
        f_X = self.forwardFCNN(X)

        # Lyapunov        
        V = self.forwardLyapunov(X) # (b)
        dV = self.gradient_lyapunov(X) # (b x n)

        # best approx. of f_X that ensures stability
        f_opt = self.fOpt(f_X, V, dV)

        return f_opt

    def forwardFCNN(self, X):
        return self.fcnn_fc1(X)

    def forwardLyapunov(self, X):
        # ICNN
        g_X = self.forwardICNN(X)
        with torch.no_grad():
            g_zero = self.forwardICNN(torch.zeros(X.shape)) # TODO: optimize    

        sigma_lyap = self.relu(torch.flatten(g_X) - torch.flatten(g_zero))
        V = sigma_lyap + self.epsilon*torch.diagonal(X@X.T)
        return V

    def forwardICNN(self, X):
        x_icnn_fc1 = self.icnn_fc1(X)
        x_icnn_sp1 = self.sp(x_icnn_fc1)

        x_icnn_fc2 = self.icnn_fc2(x_icnn_sp1)
        x_icnn_im2 = self.icnn_im2(X)
        x_icnn_sp2 = self.sp(x_icnn_fc2 + x_icnn_im2)

        x_icnn_fc3 = self.icnn_fc3(x_icnn_sp2)
        x_icnn_im3 = self.icnn_im3(X)
        g_X = self.sp(x_icnn_fc3 + x_icnn_im3)

        return g_X

    def gradient_lyapunov(self, X):
        """
        Description: calc. gradient of lyapunov fct. V (b x n)
        In X: input batch (b x n)
        Out dV: gradient of lyapunov fct. V (b x n)
        """
        dV = torch.autograd.functional.jacobian(self.forwardLyapunov, X)
        dV = torch.diagonal(dV,dim1=0,dim2=1).permute(1,0)
        return dV

    def fOpt(self, f_X, V, dV):
        stability_conditions = torch.diagonal(dV@f_X.T) + self.alpha*V # (b)
        dV_norm = (dV.T/torch.sum(dV*dV, dim=1)).T
        f_opt = f_X - (dV_norm.T*self.relu(stability_conditions)).T

        return f_opt


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
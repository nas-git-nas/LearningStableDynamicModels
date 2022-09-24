import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class FCNN(nn.Module):
    def __init__(self, n_input=10, n_hidden1=100, n_out=1):
        super(FCNN, self).__init__()

        # fully connected layers
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_out)
 
        # non-linear output fcts.
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x_in):
        # first layer
        x_fc1 = self.fc1(x_in)
        x_relu = self.relu(x_fc1)

        # second layer
        x_fc2 = self.fc2(x_relu)
        x_sig = self.sig(x_fc2)
        return x_sig

    def predict(self, x):
        logits = self.forward(x)
        return nn.Softmax(logits)


def testFCNN():
    # define parameters
    n_input = 4
    n_hidden1 = 100
    n_out = 1
    batch_size = 100
    learning_rate = 0.01
    n_epochs = 10000

    # create random data for testing
    data_x = torch.randn(batch_size, n_input)
    data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()

    # define model, loss and optimizer
    model = FCNN(n_input, n_hidden1, n_out)
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
    testFCNN()
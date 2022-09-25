import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from damped_harmonic_oscillator import DampedHarmonicOscillator

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        # model parameters
        n_input = 2
        n_hidden1 = 60
        n_out = 2

        # fully connected layers
        self.fc1 = nn.Linear(n_input, n_out, bias=False)
        # self.fc2 = nn.Linear(n_hidden1, n_out)
 
        # non-linear output fcts.
        # self.relu = nn.ReLU()
        # self.sig = nn.Sigmoid()

    def forward(self, x_in):
        # first layer
        x_fc1 = self.fc1(x_in)
        # x_relu = self.relu(x_fc1)

        # second layer
        # x_fc2 = self.fc2(x_fc1)
        # x_sig = self.sig(x_fc2)
        return x_fc1

    def predict(self, x):
        return self.forward(x)


def testFCNN():
    # learning parameters
    batch_size = 100
    learning_rate = 0.01
    nb_epochs = 1000
    nb_batches = 100

    # create random data for testing
    # data_x = torch.randn(batch_size, 2)
    # data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()
    dho = DampedHarmonicOscillator()
    data_x, data_y = dho.generate_batch(batch_size*nb_batches)

    # define model, loss and optimizer
    model = FCNN()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # run n_epochs optimizations
    losses = []
    for _ in range(nb_epochs):
        for i in range(nb_batches):
            batch_x = data_x[i:(i+1)*batch_size,:]
            batch_y = data_y[i:(i+1)*batch_size,:]

            
            pred_y = model.forward(batch_x)
            loss = loss_function(pred_y, batch_y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # test model on test data set
    test_x, test_y = dho.generate_batch(batch_size)
    pred_y = model.forward(test_x)
    loss = loss_function(pred_y, test_y)
    print(f"Error on test data set is = {loss}")

    for param in model.parameters():
        print(param.data)

    # plot resulting loss
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()



if __name__ == "__main__":
    testFCNN()
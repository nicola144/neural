import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns


# Arguments
def ask():
    args = dict()

    args['n_inputs'] = int(input("Number of inputs: "))
    args['n_neurons'] = int(input("Number of hidden neurons: "))
    isnoise = input("Want noise? Return for yes")
    args['noise'] = True if isnoise == '' else False
    return args


# Hyperparameters
EPOCHS = 500
learning_rate = 1e-4

# Model
class NN(nn.Module):

    def __init__(self, hidden_neurons):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2, hidden_neurons, True)
        self.fc2 = nn.Linear(hidden_neurons, 1, True)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Weight initialization
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)

# Adding noise
def addnoise(data):
    data = np.asarray(data)
    noise = np.random.normal(0, 0.25, data.shape)
    newdata = data + noise
    return newdata

# Root mean squared error
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


if __name__=="__main__":

    # Get settings
    args = ask()

    net = NN(hidden_neurons=args['n_neurons'])

    # Number of inputs (for each type)
    n = args['n_inputs']

    # Generate data
    data = [[0,0]] * n + [[0,1]] * n + [[1,0]] * n + [[1,1]] * n

    # add noise
    if(args['noise']):
        print("using noise")
        data = addnoise(data)

    # Generate labels
    labels = [[0]] * n + [[1]] * n + [[1]] * n + [[0]] * n

    inputs = list(map(lambda s: Variable(torch.Tensor([s])), data))
    targets = list(map(lambda s: Variable(torch.Tensor([s])), labels))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # criterion = RMSELoss
    # optimizer = optim.SGD(net.parameters(), lr=0.05)
    hold_loss=[]

    for epoch in range(0, EPOCHS):
        running_loss = 0.0
        for input, target in zip(inputs, targets):
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(input)
            loss = criterion(output, target)
            running_loss += loss.data
            loss.backward()
            optimizer.step()    # Does the update
        hold_loss.append(running_loss)

    print("\nFinal results:")
    plt.plot(np.array(hold_loss))
    plt.show()

    # Other stuff
    X = torch.Tensor([[0,0],[0,1], [1,0], [1,1]])

    model_params = list(net.parameters())

    model_weights = model_params[0].data.numpy()
    model_bias = model_params[1].data.numpy()

    plt.scatter(X.numpy()[[0,-1], 0], X.numpy()[[0, -1], 1], s=50)
    plt.scatter(X.numpy()[[1,2], 0], X.numpy()[[1, 2], 1], c='red', s=50)

    x_1 = np.arange(-0.1, 1.1, 0.1)
    y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
    plt.plot(x_1, y_1)

    x_2 = np.arange(-0.1, 1.1, 0.1)
    y_2 = ((x_2 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
    plt.plot(x_2, y_2)

    plt.legend(["neuron_1", "neuron_2"], loc=8)
    plt.show()

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime

from progress.bar import Bar

# Arguments
def ask():
    args = dict()
    args['n_inputs'] = int(input("Number of inputs(per type): "))
    args['n_neurons'] = int(input("Number of hidden neurons: "))
    isnoise = input("Want noise? Return for yes ")
    args['noise'] = True if isnoise == '' else False
    if(not (args['noise']) ):
        args['noise_val'] = 0.0
    return args

# Hyperparameters
EPOCHS = 10000
learning_rate = 1e-2

# Generate an XOR dataset. Each input gets a fair representation
def generate_data(args):

    # Number of inputs (for each type)
    n = args['n_inputs']

    # Generate data
    data = [[0,0]] * n + [[0,1]] * n + [[1,0]] * n + [[1,1]] * n

    # add noise
    if(args['noise']):
        data,args = addnoise(data,args)
    else:
        print("Not using noise")

    # Generate labels
    labels = [[0]] * n + [[1]] * n + [[1]] * n + [[0]] * n

    # Ask if they want to plot dataset
    wanna_plot = input("Wanna plot data? Return for yes ")
    wanna_plot = True if wanna_plot == '' else False
    if(wanna_plot):
        plot_data(data,labels)

    inputs = list(map(lambda s: Variable(torch.Tensor([s])), data))
    targets = list(map(lambda s: Variable(torch.Tensor([s])), labels))

    return inputs,targets,data,labels

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
def addnoise(data, args):
    data = np.asarray(data)
    noise_val = 0.25
    noise = np.random.normal(0, noise_val, data.shape)
    args['noise_val'] = noise_val
    print("Using noise: ", noise_val)
    newdata = data + noise
    return newdata, args

# Plotting dataset
def plot_data(data,labels):
    data = np.asarray(data)
    labels = np.asarray(labels)
    labels = labels.flatten()

    plt.figure(figsize=(12,8))
    plt.scatter(data[:,0], data[:,1], c=labels)
    plt.show()
    plt.close()

if __name__ == "__main__":

    # Get settings
    args = ask()

    # Model
    net = NN(hidden_neurons=args['n_neurons'])

    # Initialize weights
    net.apply(weights_init)

    # Train data
    inputs,targets,data,labels = generate_data(args)

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    hold_loss=[]

    prog_bar = Bar('Training...', suffix='%(percent).1f%% - %(eta)ds - %(index)d / %(max)d', max=EPOCHS )

    # Train loop
    for epoch in range(0, EPOCHS):
        running_loss = 0.0

        # Batch gradient descent
        inputs = np.asarray(data)
        targets = np.asarray(labels)
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(inputs)
        loss = criterion(output, targets)
        running_loss += loss.data
        loss.backward()
        optimizer.step()    # Does the update

        # Stochastic gradient descent (on each training example)
        # for input, target in zip(inputs, targets):
        #     print(type(input))
        #     print(type(target))
        #     sys.exit()
        #     optimizer.zero_grad()   # zero the gradient buffers
        #     output = net(input)
        #     loss = criterion(output, target)
        #     running_loss += loss.data
        #     loss.backward()
        #     optimizer.step()    # Does the update

        hold_loss.append(running_loss)
        prog_bar.next()

    # Results
    print("\nFinal results:")
    plt.plot(np.array(hold_loss))
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.savefig("./train_loss/train_loss_at_"+str(datetime.datetime.now())+"_model_("+str(args['n_inputs'])+','+str(args['n_neurons'])+','+str(args['noise_val'])+").png")
    plt.close()

    # Plotting decision boundary

    # _,_,data,labels = generate_data(args)

    test_data = torch.FloatTensor(data)
    labels = np.asarray(labels)
    Y_test = labels.flatten()
    y_hat_test = net(test_data)
    y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
    test_accuracy = np.sum(Y_test.reshape(-1,1)==y_hat_test_class) / len(Y_test)
    print("Accuracy {:.2f}".format(test_accuracy))

    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = test_data[:, 0].min()-0.1, test_data[:, 0].max()+0.1
    y_min, y_max = test_data[:, 1].min()-0.1, test_data[:, 1].max()+0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    full_data = np.hstack((XX.ravel().reshape(-1,1), YY.ravel().reshape(-1,1)))

    # Pass data to predict method
    data_t = torch.FloatTensor(full_data)
    db_prob = net(data_t)

    clf = np.where(db_prob<0.5,0,1)

    Z = clf.reshape(XX.shape)

    plt.figure(figsize=(12,8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
    plt.scatter(test_data[:,0], test_data[:,1], c=Y_test, cmap=plt.cm.Accent)
    plt.show()
    plt.savefig("./boundaries/decision_boundary_at_"+str(datetime.datetime.now())+"_model_("+str(args['n_inputs'])+','+str(args['n_neurons'])+','+str(args['noise_val'])+ ', lr='+str(learning_rate)+ ', epochs=' + str(EPOCHS) + ").png")
    plt.close()

# Notes

# 32 inputs (per category) 4 hidd neurons takes 1000 epochs to learn , 500 is not enough
# Weight initialization : performance from 0.5 to 0.92 (128 inputs, 8 hidd neurons, 1000 epochs, SGD no momentum)
# SGD learning rate 1e-4 was too low. Works with 1e-2

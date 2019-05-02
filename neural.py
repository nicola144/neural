import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import datetime
import math
import pandas as pd
# from progress.bar import Bar
import time
from network import NeuralNet
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Hyperparameters
# 5000 epochs work
K = 5e6
learning_rate = 0.1

# Arguments
def ask():
    args = dict()
    args['n_inputs'] = int(input("Number of inputs(per type): "))
    args['n_neurons'] = int(input("Number of hidden neurons: "))
    # isnoise = input("Want noise? Return for yes ")
    # args['noise'] = True if isnoise == '' else False
    args['noise'] = True ## NOTE:
    if(not (args['noise']) ):
        args['noise_val'] = 0.0
    return args

# Generate an XOR dataset. Each input gets a fair representation
def new_generate_data(istest):

    n = 16

    # Generate data
    noiseless_data = [[0,0]] * n + [[0,1]] * n + [[1,0]] * n + [[1,1]] * n

    # Generate labels
    labels = [[0]] * n + [[1]] * n + [[1]] * n + [[0]] * n

    data = addnoise(noiseless_data)


    # Ask if they want to plot dataset
    # wanna_plot = input("Wanna plot data? Return for yes ")
    # wanna_plot = True if wanna_plot == '' else False
    # if(wanna_plot):
    #     plot_data(data,labels,istest)


    inputs = np.asarray(data)
    targets = np.asarray(labels)
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float()

    return inputs,targets,data,labels

# Adding noise
def addnoise(data):
    data = np.asarray(data)
    noise_val = 0.5
    # Gaussian noise with 0 mean
    noise = np.random.normal(0, noise_val, data.shape)
    newdata = data + noise
    return newdata

# Plotting dataset
def plot_data(data,labels,istest):
    data = np.asarray(data)
    labels = np.asarray(labels)
    labels = labels.flatten()

    plt.figure(figsize=(12,8))
    plt.scatter(data[:,0], data[:,1], c=labels)
    if(istest):
        plt.savefig("./train.png")
    else:
        plt.savefig("./test.png")
    plt.close()

def print_results(hold_loss, args):
    print("\nFinal results:")
    print("Final MSE error: ", hold_loss[-1])
    plt.plot(np.array(hold_loss))
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.savefig("./train_loss/train_loss_at_"+str(datetime.datetime.now())+"_model_("+str(args['n_inputs'])+','+str(args['n_neurons'])+").png")
    plt.close()

    with open("results.txt", "a") as f:
        towrite = "\n----------------------------------\n Evaluating network with "+ str(args['n_inputs'])+ " inputs (of each type) and "+ str(args['n_neurons'])+ " hidden neurons, \t with K and lr = "+ str(K) + ",  " + str(learning_rate)  +"\n train MSE "+ str(hold_loss[-1]) + "\n"
        f.write(towrite)

def print_accuracy(data_test, Y_test, args):
    y_hat_test = net(data_test)
    y_hat_test_class = np.where(y_hat_test.detach().numpy()<0.5, 0, 1)
    test_accuracy = np.sum(Y_test.reshape(-1,1)==y_hat_test_class) / len(Y_test)
    print("Test Accuracy {:.2f}".format(test_accuracy))
    with open("results.txt", "a") as f:
        towrite = "Test accuracy: "+ str(test_accuracy) + "\n ---------------------------------------"
        f.write(towrite)

def plot_decision(data_test, net, args, Y_test):
    # Determine grid range in x and y directions
    x_min, x_max = data_test[:, 0].min()-0.1, data_test[:, 0].max()+0.1
    y_min, y_max = data_test[:, 1].min()-0.1, data_test[:, 1].max()+0.1

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    full_data_test = np.hstack((XX.ravel().reshape(-1,1), YY.ravel().reshape(-1,1)))

    # Pass data to predict method
    data_tensor = torch.FloatTensor(full_data_test)
    db_prob = net(data_tensor)

    clf = np.where(db_prob<0.5,0,1)

    Z = clf.reshape(XX.shape)

    plt.figure(figsize=(12,8))
    plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
    plt.scatter(data_test[:,0], data_test[:,1], c=Y_test, cmap=plt.cm.Accent)
    plt.savefig("./boundaries/decision_boundary_at_"+str(datetime.datetime.now())+"_model_("+str(args['n_inputs'])+','+str(args['n_neurons'])+ ', lr='+str(learning_rate)+ ', epochs=' + str(EPOCHS) + ").png")
    # plt.show()
    plt.close()

# Heatmap adapted from Jacob Taylor
def plot_heatmap(net, args):
    unitSquareMap = {"x":[],"y":[]}
    # Lower bound of the axes
    lo = 0
    # Upper bound of the axes
    hi = 1
    # Increase this for higher resolution map. E.g 5 means 5x5 output map
    ssf = 1000
    # Creates dictionary of grid of float coordinates.
    for i in range(0,ssf+1):
        for j in range(0,ssf+1):
            unitSquareMap["x"].append((float(hi-lo)/ssf)*i+lo)
            unitSquareMap["y"].append((float(hi-lo)/ssf)*j+lo)

    unitSquareMap = pd.DataFrame(data=unitSquareMap)
    unitSquareMap = unitSquareMap[["x","y"]]
    unitSquareMap = unitSquareMap.values

    out2 = net(torch.FloatTensor(unitSquareMap))

    inp = unitSquareMap
    inp = inp[:,:2]

    # Empty square array
    outImg = np.empty((ssf+1,ssf+1))

    for i in range(0,(ssf+1)**2):
        row = inp[i]
        outImg[int(row[0]*ssf),int(row[1]*ssf)] = out2[i]

    outImg = outImg.T
    # This is prints in the same layout as the graph
    # print(np.flip(outImg,axis=0))

    img = plt.imshow(outImg,cmap="Greys",interpolation='none',extent = [lo,hi,hi,lo])
    plt.xlabel("V1")
    plt.ylabel("V2")
    # Flip vertically, as the plot plots from the top by default
    plt.gca().invert_yaxis()
    plt.savefig("./heatmaps/heatmaps_at_"+str(datetime.datetime.now())+"_model_("+str(args['n_inputs'])+','+str(args['n_neurons'])+','+ ', lr='+str(learning_rate)+ ', epochs=' + str(EPOCHS) + ").png")
    # plt.show()
    plt.close()

def plot_roc(data_test, net, args, Y_test):
    data_test = torch.FloatTensor(data_test)

    pred = net(data_test)
    pred = pred.detach().numpy()
    test = Y_test

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(test, pred)
        roc_auc[i] = auc(fpr[i], tpr[i])
    print("ROC AUC score", roc_auc_score(test, pred))
    plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig("./roc_curves/roc_at_"+str(datetime.datetime.now())+"_model_("+str(args['n_inputs'])+','+str(args['n_neurons'])+', '+ ', lr='+str(learning_rate)+ ', epochs=' + str(EPOCHS) + ").png")
    plt.close()

def print_test_mse(data_test, labels_test, net):
    preds = net(data_test)
    loss = criterion(preds, labels_test)
    with open("results.txt", "a") as f:
        towrite = "Test MSE : "+ str(loss) + "\n ---------------------------------------"
        f.write(towrite)


if __name__ == "__main__":

    with open("results.txt", "a") as f:
        towrite = "\n ---------------------------------------- BEGIN TEST ---------------------------------------\n"
        f.write(towrite)

    # Train data
    inputs,targets,data,labels = new_generate_data(istest=False)

    # Get test data. Has to be 64 inputs (ie 16 per type) generated with a different noise from trainining set of course
    inputs_test,targets_test,data_test,labels_test = new_generate_data(istest=True)

    # (0,0)s : type 1
    type1 = inputs[0:16]
    type1_targets = targets[0:16]

    # (0,1): type 2
    type2 = inputs[16:32]
    type2_targets = targets[16:32]

    # (1,0) type 3
    type3 = inputs[32:48]
    type3_targets = targets[32:48]

    # (1,1) type 4
    type4 = inputs[48:64]
    type4_targets = targets[48:64]

    # This is training for all the required settings in the coursework specification
    for n_neurons in [2,4,8]:
        for n_inputs in [4,8,16]:

            from_type1 = type1[0:n_inputs]
            from_type1_targets = type1_targets[0:n_inputs]

            from_type2 = type2[0:n_inputs]
            from_type2_targets = type2_targets[0:n_inputs]

            from_type3 = type3[0:n_inputs]
            from_type3_targets = type3_targets[0:n_inputs]

            from_type4 = type4[0:n_inputs]
            from_type4_targets = type4_targets[0:n_inputs]

            inputs = torch.cat((from_type1, from_type2, from_type3, from_type4), 0 )

            targets = torch.cat( (from_type1_targets,from_type2_targets,from_type3_targets,from_type4_targets)  ,0)

            args = dict()
            args['n_inputs'] = n_inputs
            args['n_neurons'] = n_neurons

            # Model
            net = NeuralNet(hidden_neurons=args['n_neurons'])

            criterion = nn.MSELoss(reduction="mean")
            optimizer = optim.SGD(net.parameters(), lr=learning_rate)
            hold_loss=[]

            EPOCHS = math.ceil(K /(n_inputs * 4))
            # prog_bar = Bar('Training...', suffix='%(percent).1f%% - %(eta)ds - %(index)d / %(max)d', max=EPOCHS )
            # Train loop
            for epoch in range(0, EPOCHS):
                running_loss = 0.0

                # Batch gradient descent
                optimizer.zero_grad()
                output = net(inputs)
                loss = criterion(output, targets)
                running_loss += loss.data
                loss.backward()
                optimizer.step()

                # Printing current loss each 100 epochs
                if(epoch % 100 == 0):
                    print("\nEpoch: ", epoch, " Loss: ", running_loss.item())

                # Stochastic gradient descent (update on each training example)
                # for input, target in zip(inputs, targets):
                #     optimizer.zero_grad()   # zero the gradient buffers
                #     output = net(input)
                #     loss = criterion(output, target)
                #     running_loss += loss.data
                #     loss.backward()
                #     optimizer.step()    # Does the update

                hold_loss.append(running_loss)
                # prog_bar.next()

            # Results
            print_results(hold_loss, args)

            print_test_mse(inputs_test, targets_test,net)

            data_test = torch.FloatTensor(data_test)
            labels_test = np.asarray(labels_test)
            Y_test = labels_test.flatten()

            # Accuracy as fraction of misclassified points
            print("Evaluating model with", args['n_inputs'], " inputs per types and ", args['n_neurons'], "hidden neurons")
            print_accuracy(data_test, Y_test, args)
            time.sleep(3)

            # Plot the decision boundary
            plot_decision(data_test, net, args, Y_test)

            # Plot receiver operating characteristic curve
            plot_roc(data_test, net, args, Y_test)

            # Plot heatmap
            plot_heatmap(net, args)

    with open("results.txt", "a") as f:
        towrite = "\n ---------------------------------------- END OF TEST ---------------------------------------\n"
        f.write(towrite)

# Notes
# need to use inputs 4, 8, 16.  neurons 2,4,8
# 32 inputs (per category) 4 hidd neurons takes 1000 epochs to learn , 500 is not enough
# Weight initialization : performance from 0.5 to 0.92 (128 inputs, 8 hidd neurons, 1000 epochs, SGD no momentum)
# SGD learning rate 1e-4 was too low. Works with 1e-2
# 1 is black

# 5000 , sgd with 1e-2 16 inputs and 2 hideen neurons does not learn well ie acc 0.62 on test set. 10000 is even worse. Different run gets .73
# Surprisingly learns with 1e5 epochs .95

# unstable with 2 inps 2 neurons .5 lr 5000 epochs, sometimes learns, sometimes does not

# K = 20000
# ie , n epochs = 1250, 625, 313
# K = 100000
# ie , n epochs = 3125,

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import sys

# Model
class NeuralNet(nn.Module):

    def __init__(self, hidden_neurons):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(2, hidden_neurons, True)
        self.fc2 = nn.Linear(hidden_neurons, 1, True)
        # Weight initialization
        # init.xavier_uniform_(self.fc1.weight, gain=1)
        # init.xavier_uniform_(self.fc2.weight, gain=1)
        init.normal_(self.fc1.weight, mean=0, std=1)
        init.normal_(self.fc2.weight, mean=0, std=1)
        init.constant_(self.fc1.bias, -1)
        init.constant_(self.fc2.bias, -1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

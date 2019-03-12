import torch
from torch.autograd import Variable
import torch.nn as nn

# Model
class NeuralNet(nn.Module):

    def __init__(self, hidden_neurons):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(2, hidden_neurons, True)
        self.fc2 = nn.Linear(hidden_neurons, 1, True)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

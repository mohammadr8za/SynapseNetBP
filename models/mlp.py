import os
import torch
import torch.nn as nn


class BPNeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BPNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 700)
        self.fc2 = nn.Linear(700, 700)
        self.fc3 = nn.Linear(700, 700)
        self.fc4 = nn.Linear(700, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid( self.fc4(x))
        return x


# if __name__ == "__main__":
#     pass

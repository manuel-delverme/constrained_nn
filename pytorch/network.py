import torch
from torch import nn as nn
from torch.nn import functional as F


class ConstrNetwork(nn.Module):
    def __init__(self, dataset_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.states = nn.ParameterList([
            nn.Parameter(torch.zeros(dataset_size, 9216), requires_grad=True),
            nn.Parameter(torch.zeros(dataset_size, 128), requires_grad=True),
        ])
        self.multipliers = nn.ParameterList([
            nn.Parameter(torch.zeros(dataset_size, 9216), requires_grad=True),
            nn.Parameter(torch.zeros(dataset_size, 128), requires_grad=True),
        ])

    def step(self, x0, indices):
        x1, x2 = self.states
        return (
            self.block1(x0),
            self.block2(x1[indices]),
            self.block3(x2[indices])
        )

    def forward(self, x0, indices):
        x = self.step(x0, indices)
        states = [x[indices] for x in self.states]
        multi = [l[indices] for l in self.multipliers]

        x_T = x[-1]
        x_i = x[:-1]
        h = [a - b for a, b in zip(x[:-1], states)]
        rhs = torch.stack([torch.sum(a * b) for a, b in zip(x_i, multi)])
        return x_T, h, torch.sum(rhs), (x_i, multi)

    def shooting_forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def block3(self, x):
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def block2(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x

    def block1(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

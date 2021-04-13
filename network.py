import torch
import tqdm
from torch import nn as nn
from torch.nn import functional as F

import config

EPS = 1e-9


class ConstrNetwork(nn.Module):
    def __init__(self, train_loader):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        dataset_size = len(train_loader.dataset)
        weight = torch.zeros(dataset_size, 128)

        self.weights = nn.Sequential(
            nn.Embedding(dataset_size, 1, sparse=True),
            nn.Sigmoid(),
        )
        self.multipliers = nn.Sequential(
            nn.Embedding(dataset_size, 1, _weight=torch.ones(dataset_size, 1), sparse=True),
            # nn.ReLU() # PL suggests forcing the multipliers to R+ only during forward pass (but not backward)
            # Im not sure about the lack of backward
            nn.Softplus(),
        )

    def forward(self, x0, indices):
        dataset_weights = self.weights(indices)

        h = self.block1(x0)
        y_hat = self.block3(h)

        p_data_used = 1. - dataset_weights.mean()
        prob_defect = torch.relu(p_data_used - config.chance_constraint)
        defect = prob_defect.repeat(dataset_weights.shape)

        return y_hat, dataset_weights, defect

    def full_rollout(self, x):
        x = self.block1(x)
        # x = self.block2(x)
        x = self.block3(x)
        return x

    def block3(self, x):
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def block2(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        return x

    def block1(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.block2(x)
        return x

import torch
import tqdm
from torch import nn as nn
from torch.nn import functional as F

import config

EPS = 1e-9


def smooth_epsilon_insensitive(x, eps, tau=10):
    return x * (torch.tanh((x / eps) ** tau))


class ConstrNetwork(nn.Module):
    def __init__(self, train_loader):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        dataset_size = len(train_loader.dataset)
        weight = torch.zeros(dataset_size, 128)

        if config.initial_forward:
            with torch.no_grad():
                for batch_idx, (data, target, indices) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                    x_i = self.block1(data)
                    weight[indices] = x_i

        self.x1 = nn.Sequential(
            nn.Embedding(dataset_size, 128, _weight=weight, sparse=True),
            nn.ReLU()
        )
        self.multipliers = nn.Sequential(
            nn.Embedding(dataset_size, 128, _weight=torch.zeros(dataset_size, 128), sparse=True),
        )

    def forward(self, x0, indices):
        x1_target = self.x1(indices)

        x1_hat = self.block1(x0)
        x_T = self.block3(x1_target)

        h = x1_hat - x1_target

        eps_h = F.softshrink(h, config.constr_margin)
        defect = eps_h
        return x_T, defect

    def full_rollout(self, x):
        x = self.block1(x)
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
        x = torch.flatten(x, 1)
        x = self.block2(x)
        return x

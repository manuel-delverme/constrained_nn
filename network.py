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
            # nn.ReLU() # PL suggests forcing the multipliers to R+ only during forward pass (but not backward)
            # Im not sure about the lack of backward
            # nn.Softplus(),
            nn.Sigmoid(),
        )

    def step(self, x0, states):
        # x1, x2 = self.states
        x2, = states
        return (
            self.block1(x0),
            # self.block2(x1),
            self.block3(x2)
        )

    def forward(self, x0, indices):
        x1_target = self.x1(indices)

        x1_hat = self.block1(x0)
        x_T = self.block3(x1_target)

        h = x1_hat - x1_target

        # eps_h = h.abs() - config.constr_margin
        eps_h = smooth_epsilon_insensitive(h, config.constr_margin)
        # eps_defect = torch.relu(eps_h)

        if isinstance(config.chance_constraint, float):
            broken_constr_prob = torch.tanh(eps_h.abs()).mean()
            prob_defect = broken_constr_prob - config.chance_constraint
            defect = prob_defect.repeat(eps_h.shape)
        else:
            defect = eps_h

        return x_T, defect

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

import torch
import tqdm
from torch import nn as nn
from torch.nn import functional as F

import config


class TargetPropNetwork(nn.Module):
    def __init__(self, train_loader):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

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
        return x_T, eps_h

    def full_rollout(self, x):
        x = self.block1(x)
        x = self.block3(x)
        return x


class CIFAR10Net(nn.Module):
    def __init__(self, train_loader):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),

        )
        self.block3 = nn.Sequential(
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1)
        )

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
        return x_T, eps_h

    def full_rollout(self, x):
        x = self.block1(x)
        x = self.block3(x)
        return x

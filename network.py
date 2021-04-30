import torch
import tqdm
from torch import nn as nn
from torch.nn import functional as F

import config


class TargetPropNetwork(nn.Module):
    def __init__(self, train_loader=None, multi_stage=True):
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
        if multi_stage:
            if config.distributional:
                dataset_size = len(train_loader.dataset)
                num_classes = len(train_loader.dataset.classes)
                self.targets = train_loader.dataset.targets
                self.means = nn.Parameter(torch.zeros(num_classes, 128))
                self.scale = nn.Parameter(torch.ones(num_classes, 128))
                self.x1 = torch.distributions.Normal(self.means, self.scale)
                self.multipliers = nn.Sequential(
                    nn.Embedding(dataset_size, 128, _weight=torch.zeros(dataset_size, 128), sparse=True),
                )
            else:
                dataset_size = len(train_loader.dataset)
                weight = torch.zeros(dataset_size, 128)

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
        x1_hat = self.block1(x0)

        if config.distributional:
            targets = self.targets[indices]
            h = (x1_hat - self.x1.mean[targets]) / self.x1.scale[targets]
            samples = self.x1.rsample((x0.shape[0],))
            x1 = []
            for sample, target in zip(samples, targets):
                x1.append(sample[target])
                # x1.append(self.x1.mean[target])
            x1_target = torch.stack(x1)
        else:
            x1_target = self.x1(indices)
            h = x1_hat - x1_target

        eps_h = F.softshrink(h, config.constr_margin)

        x_T = self.block3(x1_target)
        return x_T, eps_h

    def full_rollout(self, x):
        x = self.block1(x)
        x = self.block3(x)
        return x


class CIAR10TargetProp(nn.Module):
    def __init__(self, train_loader=None, multi_stage=True):
        super().__init__()

        state_size = 84
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, state_size),

        )
        self.block3 = nn.Sequential(
            nn.Linear(state_size, 10),
            nn.LogSoftmax(dim=1)
        )

        if multi_stage:
            dataset_size = len(train_loader.dataset)
            weight = torch.zeros(dataset_size, state_size)

            if config.initial_forward:
                with torch.no_grad():
                    for batch_idx, (data, target, indices) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                        x_i = self.block1(data)
                        weight[indices] = x_i

            self.x1 = nn.Sequential(
                nn.Embedding(dataset_size, state_size, _weight=weight, sparse=True),
                nn.ReLU()
            )
            self.multipliers = nn.Sequential(
                nn.Embedding(dataset_size, state_size, _weight=torch.zeros(dataset_size, state_size), sparse=True),
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


class MNISTNetwork(TargetPropNetwork):
    def __init__(self):
        super().__init__(multi_stage=False)

    def forward(self, x0, indices):
        return self.full_rollout(x0)


class CIFAR10Network(CIAR10TargetProp):
    def __init__(self):
        super().__init__(multi_stage=False)

    def forward(self, x0, indices):
        return self.full_rollout(x0)


class MNISTLiftedNetwork(TargetPropNetwork):
    def __init__(self, num_constraints):
        super().__init__(multi_stage=False)
        self.x1 = nn.Sequential(
            nn.Embedding(num_constraints, 1, sparse=True),
            nn.ReLU()
        )
        self.multipliers = nn.Sequential(
            nn.Embedding(num_constraints, 1, sparse=True),
        )

    def forward(self, x0, indices):
        return self.full_rollout(x0)


class CIFAR10LiftedNetwork(CIAR10TargetProp):
    def __init__(self, num_constraints):
        super().__init__(multi_stage=False)
        self.x1 = nn.Sequential(
            nn.Embedding(num_constraints, 1, sparse=True),
            nn.ReLU()
        )
        self.multipliers = nn.Sequential(
            nn.Embedding(num_constraints, 1, sparse=True),
        )

    def forward(self, x0, indices):
        return self.full_rollout(x0)

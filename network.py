import torch
from torch import nn as nn

import config


class SplitNet(nn.Module):
    def __init__(self, dataset="mnist"):
        super().__init__()
        if dataset == "mnist":
            self.blocks = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(1, 32, 3, 1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(1),
                    nn.Linear(9216, 128),
                    nn.Identity() if config.distributional else nn.ReLU(),
                ),
                *([nn.Sequential(
                    nn.Linear(128, 128),
                    nn.Identity() if config.distributional else nn.ReLU(),
                )] * config.num_layers),
                nn.Sequential(
                    nn.Linear(128, 10),
                    nn.LogSoftmax(dim=1)
                ),
            )
        elif dataset == "cifar10":
            self.blocks = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 6, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(6, 16, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Flatten(1),
                    nn.Linear(16 * 5 * 5, 120),
                    nn.ReLU(),
                    nn.Linear(120, 84),
                    nn.Identity() if config.distributional else nn.ReLU(),
                ),
                *([nn.Sequential(
                    nn.Linear(84, 84),
                    nn.Identity() if config.distributional else nn.ReLU(),
                )] * config.num_layers),
                nn.Sequential(
                    nn.Linear(84, 10),
                    nn.LogSoftmax(dim=1)
                ),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.blocks(x)

    def one_step(self, x_t):
        x_t1 = [block(x_t) for x_t, block in zip(x_t, self.blocks)]
        return x_t1

    @property
    def tabular_multipliers(self):
        return self.multipliers[0]

    @property
    def distributional_multipliers(self):
        return self.multipliers[1]


class TabularStateNet(nn.Module):
    def __init__(self, dataset_size, weights, state_sizes):
        super().__init__()
        self.state_params = nn.ModuleList(TabularState(dataset_size, state_size, weight) for weight, state_size in zip(weights, state_sizes))

    def forward(self, indices):
        return [s(indices) for s in self.state_params]


class GaussianStateNet(nn.Module):
    def __init__(self, dataset_size, num_classes, state_sizes, num_samples):
        super().__init__()
        assert (config.num_samples // num_classes) > 0
        self.state_params = nn.ModuleList(GaussianState(num_classes, state_size, num_samples) for state_size in state_sizes)

    def forward(self, indices):
        xs = [state_distr(indices) for state_distr in self.state_params]
        return xs


class TabularState(nn.Module):
    def __init__(self, dataset_size, state_size, weight):
        super().__init__()
        self.state = nn.Sequential(
            nn.Embedding(dataset_size, state_size, _weight=weight, sparse=True),
            nn.ReLU()
        )

    def forward(self, indices):
        return self.state(indices)


class GaussianState(nn.Module):
    def __init__(self, num_classes, state_size, num_samples):
        super().__init__()
        self.means = nn.Linear(num_classes, state_size, bias=False)
        self.scales = nn.Sequential(
            nn.Linear(num_classes, state_size, bias=False),
            nn.Softplus(),
        )
        self.ys = torch.eye(num_classes).to(device=config.device, dtype=torch.float)
        self.num_samples = num_samples

    def forward(self, indices):
        loc, scale = self.means(self.ys), self.scales(self.ys)
        return torch.distributions.Normal(loc, scale).rsample((self.num_samples,))


class TargetPropNetwork(nn.Module):
    def __init__(self, transition: SplitNet, state_net: TabularStateNet):
        super().__init__()
        self.transition_model = transition
        self.state_model = state_net

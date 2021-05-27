import torch
import tqdm
from torch import nn as nn
from torch.nn import functional as F

import config


class TargetPropNetwork(nn.Module):
    def __init__(self, train_loader=None):
        super().__init__()
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
        dataset_size = len(train_loader.dataset)
        state_sizes = [b[0].in_features for b in self.blocks[1:]]

        if config.distributional:
            num_classes = len(train_loader.dataset.classes)
            self.state_net = GaussianStateNet(dataset_size, num_classes, state_sizes)
            self.multipliers = nn.ModuleList((
                nn.Embedding(dataset_size, state_sizes[0], _weight=torch.zeros(dataset_size, state_sizes[0]), sparse=True),
                nn.ParameterList(nn.Parameter(torch.zeros(num_classes, state_size)) for state_size in state_sizes[1:]),
            ))
        else:
            self.state_net = StateNet(dataset_size, train_loader)
            self.multipliers = nn.Sequential(
                nn.Embedding(dataset_size, 128, _weight=torch.zeros(dataset_size, 128), sparse=True),
            )

    def constrained_forward(self, x0, indices, targets):
        states = self.state_net(x0)
        activations = self.one_step(states)
        x_t, x_T = activations[:-1], activations[-1]
        defects = self.state_net.defect(x_t, targets)
        return x_T, defects

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


class StateNet(nn.Module):
    def __init__(self, dataset_size, train_loader):
        super().__init__()
        weight = torch.zeros(dataset_size, 128)

        with torch.no_grad():
            for batch_idx, (data, target, indices) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                x_i = self.block1(data)
                weight[indices] = x_i

        self.states = nn.Sequential(
            nn.Embedding(dataset_size, 128, _weight=weight, sparse=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blocks(x)

    def defect(self, activations, targets):
        raise NotImplemented
        tabular_defect = (activations[0] - self.states[0].mean[targets]) / self.states[0].scale[targets]
        defects = [F.softshrink(tabular_defect, config.tabular_margin), ]

        for a_i, target_distribution in zip(activations[1:], self.states[1:]):
            h_i = (a_i - target_distribution.mean) / target_distribution.scale
            defects.append(F.softshrink(h_i, config.distributional_margin))

        return defects


class GaussianStateNet(nn.Module):
    def __init__(self, dataset_size, num_classes, state_sizes):
        super().__init__()
        assert (config.num_samples // num_classes) > 0
        self.states = nn.ModuleList(GaussianState(num_classes, state_size) for state_size in state_sizes)

    def forward(self, x0):
        xs = [x0, ] + [s.rsample(config.num_samples) for s in self.states]
        return xs

    def defect(self, activations, targets):
        defects = [self.states[0].defect(activations[0], targets), ]
        for a_i, state in zip(activations[1:], self.states[1:]):
            defects.append(state.defect(a_i))
        return defects


class GaussianState(nn.Module):
    def __init__(self, num_classes, state_size):
        super().__init__()
        self.means = nn.Linear(num_classes, state_size, bias=False)
        self.scales = nn.Sequential(
            nn.Linear(num_classes, state_size, bias=False),
            nn.Softplus(),
        )
        self.ys = torch.eye(num_classes).to(device=config.device, dtype=torch.float)

    def forward(self):
        return self.means(self.ys), self.scales(self.ys)

    def defect(self, a_i, targets=None):
        loc, scale = self()
        h = (a_i - loc[targets]) / scale[targets]
        return F.softshrink(h, config.distributional_margin)

    def rsample(self, num_samples):
        loc, scale = self()
        return torch.distributions.Uniform(loc - scale * config.distributional_margin, loc + scale * config.distributional_margin).rsample((num_samples,))

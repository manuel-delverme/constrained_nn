import torch
import tqdm
from torch import nn as nn
from torch.nn import functional as F

import config


class TargetPropNetwork(nn.Module):
    def __init__(self, train_loader=None, dataset="mnist"):
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
            weights = [torch.zeros(dataset_size, state_size) for state_size in state_sizes]
            with torch.no_grad():
                for batch_idx, (data, target, indices) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                    a_i = data
                    for t, block in enumerate(self.blocks[:-1]):
                        a_i = block(a_i)
                        weights[t][indices] = a_i
            self.state_net = StateNet(dataset_size, weights, state_sizes)
            self.multipliers = nn.Sequential(*[nn.Embedding(dataset_size, state_size, _weight=torch.zeros(dataset_size, state_size), sparse=True) for state_size in state_sizes])

    def constrained_forward(self, x0, indices, targets):
        states = self.state_net(x0, indices)
        activations = self.one_step(states)
        x_t, x_T = activations[:-1], activations[-1]
        defects = self.state_net.defect(x_t, targets, indices)
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
    def __init__(self, dataset_size, weights, state_sizes):
        super().__init__()
        self.states = nn.ModuleList(TabularState(dataset_size, state_size, weight) for weight, state_size in zip(weights, state_sizes))

    def forward(self, x0, indices):
        return [x0, ] + [s(indices) for s in self.states]

    def defect(self, activations, targets, indices):
        defects = []
        for a_i, state in zip(activations, self.states):
            defects.append(state.defect(a_i, indices=indices))
        return defects


class TabularState(nn.Module):
    def __init__(self, dataset_size, state_size, weight):
        super().__init__()
        self.state = nn.Sequential(
            nn.Embedding(dataset_size, state_size, _weight=weight, sparse=True),
            nn.ReLU()
        )

    def forward(self, indices):
        return self.state(indices)

    def defect(self, activations, targets=None, indices=None):
        h = activations - self.state(indices)
        return F.softshrink(h, config.tabular_margin)

    def rsample(self, num_samples):
        return torch.distributions.Normal(*self()).rsample((num_samples,))


class GaussianStateNet(nn.Module):
    def __init__(self, dataset_size, num_classes, state_sizes):
        super().__init__()
        assert (config.num_samples // num_classes) > 0
        self.states = nn.ModuleList(GaussianState(num_classes, state_size) for state_size in state_sizes)

    def forward(self, x0, indices):
        xs = [x0, ] + [s.rsample(config.num_samples) for s in self.states]
        return xs

    def defect(self, activations, targets, indices):
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

    def defect(self, a_i, targets=None, indices=None):
        loc, scale = self()
        h = (a_i - loc[targets]) / scale[targets]
        return F.softshrink(h, config.distributional_margin)

    def rsample(self, num_samples):
        return torch.distributions.Normal(*self()).rsample((num_samples,))

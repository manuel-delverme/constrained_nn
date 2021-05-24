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
            ),
            nn.Sequential(
                nn.Linear(9216, 128),
                nn.Identity() if config.distributional else nn.ReLU(),
            ),
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
        # TODO: parallel
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
        defects = [(activations[0] - self.states[0].mean[targets]) / self.states[0].scale[targets]]
        # TODO: parallelize
        defects.extend([(a_i - target_distribution.mean) / target_distribution.scale for a_i, target_distribution in zip(activations[1:], self.states[1:])])
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
        loc, scale = self.states[0]()
        defects = [(activations[0] - loc[targets]) / scale[targets]]
        # TODO: parallelize
        h_distr = []
        for a_i, state in zip(activations[1:], self.states[1:]):
            loc, scale = state()
            h_distr.append((a_i - loc) / scale)
        defects.extend(h_distr)
        return defects


class GaussianState(nn.Module):
    def __init__(self, num_classes, state_size):
        super().__init__()
        self.means = nn.Linear(num_classes, state_size)
        self.scales = nn.Sequential(
            nn.Linear(num_classes, state_size),
            nn.Softplus(),
        )
        self.ys = torch.eye(num_classes).to(device=config.device, dtype=torch.float)

    def forward(self):
        return self.means(self.ys), self.scales(self.ys)

    def rsample(self, num_samples):
        return torch.distributions.Normal(*self()).rsample((num_samples,))


class CIAR10TargetProp(nn.Module):
    def __init__(self, train_loader=None, multi_stage=True):
        raise NotImplementedError
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
            if config.distributional:
                dataset_size = len(train_loader.dataset)
                num_classes = len(train_loader.dataset.classes)
                self.target_class = torch.tensor(train_loader.dataset.targets)

                self.means = nn.Parameter(torch.rand((num_classes, state_size)))
                self.scale = nn.Parameter(torch.ones(num_classes, state_size))

                self.x1 = torch.distributions.Normal(self.means, self.scale)
                self.multipliers = nn.Sequential(
                    nn.Embedding(dataset_size, state_size, _weight=torch.zeros(dataset_size, state_size), sparse=True),
                )
            else:
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
        x1_hat = self.block1(x0)

        if config.distributional:
            targets = self.target_class[indices]
            h = (x1_hat - self.x1.mean[targets]) / self.x1.scale[targets]

            samples = self.x1.rsample((config.num_samples,))
            x1_target = samples[torch.arange(config.num_samples), targets[:config.num_samples]]
        else:
            x1_target = self.x1(indices)
            h = x1_hat - x1_target

        if config.eps_constraint:
            h2 = F.softshrink(h, config.constr_margin)
        else:
            h2 = h

        x_T = self.block3(x1_target)
        return x_T, h2

    def full_rollout(self, x):
        x = self.block1(x)
        x = self.block3(x)
        return x

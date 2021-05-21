import torch
import tqdm
from torch import nn as nn
from torch.nn import functional as F

import config


class TargetPropNetwork(nn.Module):
    def __init__(self, train_loader=None, multi_stage=True):
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
        if multi_stage:
            if config.distributional:
                dataset_size = len(train_loader.dataset)
                self.num_classes = len(train_loader.dataset.classes)
                assert (config.num_samples // self.num_classes) > 0
                self.target_class = train_loader.dataset.targets

                means = []
                sigmas = []
                multipliers = []
                state_distributions = []
                state_features = self.blocks[1][0].in_features
                # self.num_classes,
                means.append(nn.Parameter(torch.rand((self.num_classes, state_features))))
                sigmas.append(nn.Parameter(torch.ones(self.num_classes, state_features)))
                state_distributions.append(torch.distributions.Normal(means[-1], sigmas[-1], ))
                dataset_match = nn.Embedding(dataset_size, state_features, _weight=torch.zeros(dataset_size, state_features), sparse=True)

                for prev_block, next_block in zip(self.blocks[1:-1], self.blocks[2:]):
                    state_features = next_block[0].in_features
                    means.append(nn.Parameter(torch.rand((self.num_classes, state_features))))
                    sigmas.append(nn.Parameter(torch.ones(self.num_classes, state_features)))
                    state_distributions.append(torch.distributions.Normal(means[-1], sigmas[-1], ))
                    multipliers.append(nn.Parameter(torch.zeros(self.num_classes, state_features)))

                self.multipliers = nn.ModuleList((
                    dataset_match,
                    nn.ParameterList(multipliers),
                ))
                self.means = nn.ParameterList(means)
                self.scales = nn.ParameterList(sigmas)
                self.states = state_distributions
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

    @property
    def distribution_multipliers(self):
        return self.multipliers[1]

    @property
    def tabular_multipliers(self):
        return self.multipliers[0]

    def add_distributional_constraint(self, num_classes, state_features):
        self.states.append(
            torch.distributions.Normal(
                nn.Parameter(torch.rand((num_classes, state_features))),
                nn.Parameter(torch.ones(num_classes, state_features)),
            )
        )
        self.multipliers.append(
            nn.Parameter(torch.zeros(num_classes, state_features))
        )

    def constrained_forward(self, x0, indices, targets):

        if config.distributional:
            defects = []

            # Match data to the first distribution
            a_i = self.blocks[0](x0)
            h_i = (a_i - self.means[0][targets]) / self.scales[0][targets]
            defects.append(h_i)  # .flatten())

            # Match samples to the subsequent distributions
            a_i = self.states[0].rsample((config.num_samples // self.num_classes,))
            assert len(self.blocks) == len(self.states) + 1
            for layer_function, target_distribution in zip(self.blocks[1:-1], self.states[1:]):
                a_i = layer_function(a_i)

                h_i = (a_i - target_distribution.mean) / target_distribution.scale
                h_i = F.softshrink(h_i, config.constr_margin)

                defects.append(h_i)  # .flatten())
                a_i = target_distribution.rsample((config.num_samples // self.num_classes,))
            # defects = torch.cat(defects, dim=0)

        else:
            raise NotImplemented
            x1_target = self.x1(indices)
            h = x1_hat - x1_target

        a_T = self.blocks[-1](a_i)
        return a_T, defects

    def forward(self, x):
        return self.blocks(x)


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

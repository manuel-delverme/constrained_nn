import numpy as np
import torch
import torchvision.models
from torch import nn as nn


class SplitNet(nn.Module):
    def __init__(self, dataset, distributional, every_layer):
        super().__init__()
        if dataset == "mnist":
            if every_layer:
                self.blocks = nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(1, 32, 3, 1),
                        nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Conv2d(32, 64, 3, 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    ),
                    nn.Sequential(
                        nn.Flatten(start_dim=1),
                        nn.Linear(9216, 128),
                        nn.Identity() if distributional else nn.ReLU(),
                    ),
                    nn.Sequential(
                        nn.Linear(128, 10),
                        nn.LogSoftmax(dim=1)
                    ),
                )
            else:
                self.blocks = nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(1, 32, 3, 1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Flatten(1),
                        nn.Linear(9216, 128),
                        nn.Identity() if distributional else nn.ReLU(),
                    ),
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
                    nn.Identity() if distributional else nn.ReLU(),
                ),
                nn.Sequential(
                    nn.Linear(84, 10),
                    nn.LogSoftmax(dim=1)
                ),
            )
        elif dataset == "imagenet":
            alexnet = torchvision.models.alexnet()
            self.features = alexnet.features
            self.classifier = alexnet.classifier
            self.blocks = nn.Sequential(
                nn.Sequential(
                    *self.features,
                    torch.nn.Flatten(1),
                ),
                nn.Sequential(
                    *self.classifier,
                    nn.LogSoftmax(dim=1)
                )
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.blocks(x)

    def one_step(self, x_t):
        x_t1 = [block(x_t) for x_t, block in zip(x_t, self.blocks)]
        return x_t1


class TabularStateNet(nn.Module):
    def __init__(self, dataset_size, weights, state_sizes):
        super().__init__()
        self.state_params = nn.ModuleList(TabularState(dataset_size, state_size, weight) for weight, state_size in zip(weights, state_sizes))

    def forward(self, indices):
        return [s(indices) for s in self.state_params]


class GaussianStateNet(nn.Module):
    def __init__(self, dataset_size, num_classes, state_sizes, num_samples):
        super().__init__()
        self.state_params = nn.ModuleList(GaussianState(num_classes, state_size, num_samples) for state_size in state_sizes)

    def forward(self, indices):
        xs = [state_distribution(indices) for state_distribution in self.state_params]
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
        num_state_features = int(np.prod(state_size))
        self.means = nn.Sequential(
            nn.Flatten(1, ),
            nn.Linear(num_classes, num_state_features, bias=False),
            nn.Unflatten(1, state_size),
        )
        self.scales = nn.Sequential(
            nn.Flatten(1, ),
            nn.Linear(num_classes, num_state_features, bias=False),
            nn.Unflatten(1, state_size),
            nn.Softplus(),
        )
        self.ys = nn.Parameter(torch.eye(num_classes).to(dtype=torch.float), requires_grad=False)
        self.num_samples = num_samples

    def forward(self, indices):
        loc, scale = self.means(self.ys), self.scales(self.ys)
        targets = torch.distributions.Normal(loc, scale).rsample((self.num_samples,))
        return targets.reshape((targets.shape[0] * targets.shape[1], *targets.shape[2:]))


class TargetPropNetwork(nn.Module):
    def __init__(self, transition: SplitNet, state_net: TabularStateNet):
        super().__init__()
        self.transition_model = transition
        self.state_model = state_net

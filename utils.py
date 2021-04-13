import os

import torch
from torchvision import datasets

import config


class MNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.DEBUG:
            self.data, self.targets = self.data[:config.batch_size * 2], self.targets[:config.batch_size * 2]
        if config.corruption_percentage and kwargs['train']:
            num_corrupted_indices = int(config.corruption_percentage * len(self.data))
            indices = torch.randint(0, len(self.data) - 1, (num_corrupted_indices,))
            self.data[indices] = torch.randint_like(self.data[indices], self.data.max())

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


class CIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.DEBUG:
            self.data, self.targets = self.data[:config.batch_size * 2], self.targets[:config.batch_size * 2]
        if config.corruption_percentage and kwargs['train']:
            num_corrupted_indices = int(config.corruption_percentage * len(self.data))
            indices = torch.randint(0, len(self.data) - 1, (num_corrupted_indices,))
            self.data[indices] = torch.randint_like(self.data[indices], self.data.max())

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


def load_datasets(transform):
    if "SLURM_JOB_ID" in os.environ.keys():
        if config.dataset == "mnist":
            dataset1 = MNIST(config.dataset_path.format("mnist", "mnist"), train=True, transform=transform)
            dataset2 = MNIST(config.dataset_path.format("mnist", "mnist"), train=False, transform=transform)
        else:
            dataset1 = CIFAR10(config.dataset_path.format("cifar10", "cifar10"), train=True, transform=transform)
            dataset2 = CIFAR10(config.dataset_path.format("cifar10", "cifar10"), train=False, transform=transform)
    else:
        if config.dataset == "mnist":
            dataset1 = MNIST("../data", train=True, transform=transform)
            dataset2 = MNIST("../data", train=False, transform=transform)
        else:
            dataset1 = CIFAR10("../data", train=True, transform=transform)
            dataset2 = CIFAR10("../data", train=False, transform=transform)
    return dataset1, dataset2

import os

import torch
import torch.utils.data
import torchvision.transforms
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


def load_datasets():
    if config.dataset == "mnist":
        dataset_class = MNIST
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif config.dataset == "cifar10":
        dataset_class = CIFAR10
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise NotImplemented

    if "SLURM_JOB_ID" in os.environ.keys():
        dataset_path = config.dataset_path.format(config.dataset, config.dataset)
    else:
        dataset_path = "../data"

    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.batch_size * 4}
    if config.use_cuda:
        cuda_kwargs = {'num_workers': 0, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset = dataset_class(dataset_path, train=True, transform=transform)

    sampler = None
    if config.adversarial_sampling:
        sampler = torch.utils.data.WeightedRandomSampler(torch.ones(len(dataset)), num_samples=len(dataset), replacement=True)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, shuffle=True, **train_kwargs)

    test_loader = torch.utils.data.DataLoader(
        dataset_class(dataset_path, train=False, transform=transform), **test_kwargs
    )
    return train_loader, test_loader


def plot(loss, model):
    import torchviz
    import os
    torchviz.make_dot(loss, params=dict(model.named_parameters())).render("/tmp/plot.gv")
    os.system("evince /tmp/plot.gv.pdf")

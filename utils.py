import os
import subprocess

import torch
import torch.utils.data
import torchvision.transforms
from torchvision import datasets

import config


class ImageNet(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.DEBUG:
            self.samples = self.samples[:config.batch_size * 2 - 1]
            self.targets = self.targets[:config.batch_size * 2 - 1]
            self.imgs = self.imgs[:config.batch_size * 2 - 1]

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


class MNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.DEBUG:
            self.data, self.targets = self.data[:config.batch_size * 2 - 1], self.targets[:config.batch_size * 2 - 1]

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index


class CIFAR10(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if config.DEBUG:
            self.data, self.targets = self.data[:config.batch_size * 2 - 1], self.targets[:config.batch_size * 2 - 1]

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
    elif config.dataset == "imagenet":
        return load_imagenet()
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

    train_dataset = dataset_class(dataset_path, train=True, transform=transform)

    train_kwargs['shuffle'] = True
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    test_loader = torch.utils.data.DataLoader(dataset_class(dataset_path, train=False, transform=transform), **test_kwargs)
    return train_loader, test_loader


def load_imagenet():
    if "SLURM_JOB_ID" in os.environ.keys():
        dataset_home = os.path.join(os.environ["SLURM_TMPDIR"], "ImageNet")

        os.system(f"mkdir -p {dataset_home}/train {dataset_home}/val/")
        os.system(f"tar -xvf /network/datasets/imagenet/ILSVRC2012_img_train.tar -C {dataset_home}/train/")
        os.system(f"tar -xvf /network/datasets/imagenet/ILSVRC2012_img_val.tar -C {dataset_home}/val/")
        subprocess.run(
            'find ' + dataset_home + '/train/ -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done')

        os.system(f'cp -r /network/datasets/imagenet.var/imagenet_torchvision/val {dataset_home}/')
    else:
        dataset_home = "../data/ImageNet"

    train_dir = os.path.join(dataset_home, 'train')
    test_dir = os.path.join(dataset_home, 'val')
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = ImageNet(
        train_dir,
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))
    validation_dataset = ImageNet(
        test_dir,
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.data_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.data_workers, pin_memory=True)
    return train_loader, val_loader


def plot(loss, model):
    import torchviz
    import os
    torchviz.make_dot(loss, params=dict(model.named_parameters())).render("/tmp/plot.gv")
    os.system("evince /tmp/plot.gv.pdf")


def update_hyper_parameters():
    if config.distributional:
        assert config.experiment == "target-prop"

    ################################################################
    # Derivative parameters
    ################################################################
    config.device = torch.device("cuda" if config.use_cuda else "cpu")

    if config.dataset == "mnist":
        if config.constraint_satisfaction == "extra-gradient":
            if config.distributional:
                # WARNING: these are not the best hyper-parameters
                config.num_samples = 32
                config.distributional_margin = 0.3967
                config.initial_lr_theta = 0.0003638
                config.initial_lr_x = 0.05649
                config.initial_lr_y = 3.725e-07
            else:
                config.tabular_margin = 0.4373272842992752
                config.initial_lr_theta = 0.0008636215301536897
                config.initial_lr_x = 0.12499896839056827
                config.initial_lr_y = 7.270811457366213e-06
        elif config.constraint_satisfaction == "penalty":
            config.tabular_margin = 0.1017
            config.initial_lr_theta = 0.0003638
            config.initial_lr_x = 0.05649
            config.initial_lr_y = 3.725e-7
            config.lambda_ = 0.06788
            # 1e-2  # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0
        elif config.constraint_satisfaction == "descent-ascent":
            config.tabular_margin = 0.9658136136534436
            config.initial_lr_theta = 0.003314
            config.initial_lr_x = 0.04527
            config.initial_lr_y = 0.0001389
    elif config.dataset == "cifar10":
        if config.constraint_satisfaction == "extra-gradient":
            if config.distributional:
                config.distributional_margin = 0.2470519487851573
                config.initial_lr_theta = 0.004169182899797638
                config.initial_lr_x = 0.25530572068931
                config.initial_lr_y = 9.356607499463217e-07
                config.num_samples = 32
            else:
                config.tabular_margin = 0.06640108363973078
                config.initial_lr_theta = 0.003175211334723672
                config.initial_lr_x = 0.03977922031861909
                config.initial_lr_y = 2.311834855494428e-06
        elif config.constraint_satisfaction == "penalty":
            config.tabular_margin = 0.10063086881740957
            config.initial_lr_theta = 4.6397470184556474e-05
            config.initial_lr_x = 0.27691927629931706
            config.initial_lr_y = 0.004094357077137722
            config.lambda_ = 0.06788
        elif config.constraint_satisfaction == "descent-ascent":
            config.tabular_margin = 0.13308695791662822
            config.initial_lr_theta = 0.00029136889726434325
            config.initial_lr_x = 0.25009935678225476
            config.initial_lr_y = 0.00015235056347032218

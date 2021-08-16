import sys

import torch

import experiment_buddy

RUN_SWEEP = 0
REMOTE = 1

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
dataset_path = "../data" if DEBUG else "/network/datasets/{}.var/{}_torchvision"

experiment = ["sgd", "target-prop"][1]
constraint_satisfaction = ["penalty", "descent-ascent", "extra-gradient"][2]
dataset = ["mnist", "cifar10"][0]
distributional = False

batch_size = 1024

# # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0
initial_forward = True
# random_seed = 1337
random_seed = 1
num_layers = 0
warmup_lr = 0.009185

warmup_epochs = 0  # 1 if DEBUG else 0
num_epochs = 431
use_cuda = True  # not DEBUG

initial_lr_theta = 1.
initial_lr_x = 1.
initial_lr_y = 1.

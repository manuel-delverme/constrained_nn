import sys

import torch

import experiment_buddy

RUN_SWEEP = 1
REMOTE = 1

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
dataset_path = "../data" if DEBUG else "/network/datasets/{}.var/{}_torchvision"

experiment = ["sgd", "target-prop", "robust-classification"][1]
constraint_satisfaction = ["penalty", "descent-ascent", "extra-gradient"][2]
dataset = ["mnist", "cifar10"][0]
distributional = True

# Robust Classification experiments
corruption_percentage = 0.00

# Distributional
num_samples = 32

chance_constraint = {
    "sgd": False,
    "target-prop": False,
    "robust-classification": 0.05
}[experiment]

# Target Prop Experiments
distributional_margin = 0.3967
tabular_margin = 0.15779255009939092

initial_forward = True

random_seed = 1337

initial_lr_theta = 0.0003638
initial_lr_x = 0.05649
initial_lr_y = 3.725e-7
# 1e-2  # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

num_layers = 0
warmup_lr = 0.009185
lambda_ = 0.06788

batch_size = 1024
warmup_epochs = 0  # 1 if DEBUG else 0
num_epochs = 1000
use_cuda = True  # not DEBUG

################################################################
# END OF PARAMETERS
################################################################
experiment_buddy.register(locals())
device = torch.device("cuda" if use_cuda else "cpu")
if distributional:
    assert experiment == "target-prop"

################################################################
# Derivative parameters
################################################################

tb = experiment_buddy.deploy(
    host="mila" if REMOTE else "",
    sweep_yaml="sweep_hyper.yaml" if RUN_SWEEP else False,
    extra_slurm_headers="""
    """,
    proc_num=10 if RUN_SWEEP else 1
)

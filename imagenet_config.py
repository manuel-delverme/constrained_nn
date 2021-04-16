import sys

import torch

import experiment_buddy

RUN_SWEEP = 0
REMOTE = 1

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
dataset_path = "../data/imagenette2" if DEBUG else "/network/datasets/imagenet.var/imagenet_torchvision"

random_seed = 1337


dataloader_workers = 4
initial_lr_theta = .1

initial_lr_x = .05
initial_lr_y = .08
# 1e-2  # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

warmup_lr = 0.009185
lambda_ = 0.06788
# high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

batch_size = 8
warmup_epochs = 1 if DEBUG else 50
num_epochs = 150
constr_margin = 0.1
initial_forward = not DEBUG
use_cuda = not DEBUG

################################################################
# END OF PARAMETERS
################################################################
experiment_buddy.register(locals())
device = torch.device("cuda" if use_cuda else "cpu")

################################################################
# Derivative parameters
################################################################
tb = experiment_buddy.deploy(
    host="mila" if REMOTE else "",
    sweep_yaml="sweep_hyper.yaml" if RUN_SWEEP else False,
    extra_slurm_headers="""
    """,
    # SBATCH --mem=24GB
    proc_num=10 if RUN_SWEEP else 1
)
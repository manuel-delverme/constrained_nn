import sys

import torch

import experiment_buddy

RUN_SWEEP = 1
REMOTE = 1
NUM_PROCS = 10

sweep_yaml = "sweep_hyper.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

random_seed = 1337

# initial_lr_theta = .005186
# initial_lr_x = .04887
# initial_lr_y = .09284

initial_lr_theta = .001
initial_lr_x = .05
initial_lr_y = .08

# 1e-2  # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

warmup_lr = 0.009185
lambda_ = 0.06788
# high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

batch_size = 1024
warmup_epochs = 35 if DEBUG else 50
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
# esh = """
# #SBATCH --mem=24GB
# """
esh = ""
tb = experiment_buddy.deploy(host=HOST, sweep_yaml=sweep_yaml, extra_slurm_headers=esh, proc_num=NUM_PROCS)

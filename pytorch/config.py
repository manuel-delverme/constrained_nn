import sys
import torch

import mila_tools

RUN_SWEEP = 0
REMOTE = 1
NUM_PROCS = 1

sweep_yaml = "pytorch/sweep_hyper.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
DEBUG = False

random_seed = 1337

initial_lr_theta = .005186
initial_lr_x = .04887
initial_lr_y = 1.

warmup_lr = 0.009185
lambda_ = 0.06788
# high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

batch_size = 1024
warmup_epochs = 5  # 1 if DEBUG else 100  # 150
num_epochs = 50
initial_forward = not DEBUG

use_cuda = not DEBUG

################################################################
# END OF PARAMETERS
################################################################
mila_tools.register(locals())
device = torch.device("cuda" if use_cuda else "cpu")

################################################################
# Derivative parameters
################################################################
# esh = """
# #SBATCH --mem=24GB
# """
esh = ""
#SBATCH --partition=long
tb = mila_tools.deploy(host=HOST, sweep_yaml=sweep_yaml, extra_slurm_headers=esh, proc_num=NUM_PROCS)

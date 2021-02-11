import math
import sys

import mila_tools

RUN_SWEEP = 0
REMOTE = 1

sweep_yaml = "sweep_hyper.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

RANDOM_SEED = 1337

dataset = "mnist"
initial_lr_theta = .001
initial_lr_x = .05
initial_lr_y = .08
# high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

batch_size = 512

num_epochs = 100_000  # 00
eval_every = math.ceil(num_epochs / 100)

decay_steps = num_epochs  # // 4  # 500000
decay_factor = 0.7

################################################################
# END OF PARAMETERS
################################################################
mila_tools.register(locals())

################################################################
# Derivative parameters
################################################################
tb = mila_tools.deploy(host=HOST, sweep_yaml=sweep_yaml, extra_slurm_headers="""
#SBATCH --mem=24GB
""")

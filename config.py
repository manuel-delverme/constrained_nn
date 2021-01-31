import math # 1
import sys

import jax.experimental.optimizers
import mila_tools

RUN_SWEEP = 0
REMOTE = 1
# PROJECT_NAME = "constrained_nn"

sweep_yaml = "sweep_hyper.yaml" if RUN_SWEEP else False
HOST = "mila" if REMOTE else ""
DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

RANDOM_SEED = 1337

dataset = "mnist"
initial_lr_theta = .001
initial_lr_x = .05
initial_lr_y = .08
# 1e-2  # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

num_hidden = 512
blocks = [2, ] * 3
# block0 = 2
# block1 = 4
# block2 = 4
# block3 = 4

use_adam = True
grad_clip = 4.0  # avoid leaky_grad explosions
adam1 = 0.9
adam2 = 0.99

batch_size = 128
weight_norm = False  # avoid unbound targets

num_epochs = 100_000  # 00
eval_every = math.ceil(num_epochs / 100)

decay_steps = num_epochs  # // 4  # 500000
decay_factor = 1.0

################################################################
# END OF PARAMETERS
################################################################
mila_tools.register(locals())

################################################################
# Derivative parameters
################################################################
# initial_lr_y = initial_lr_x * 10.
lr_theta = jax.experimental.optimizers.inverse_time_decay(initial_lr_theta, decay_steps, decay_factor, staircase=True)
lr_x = jax.experimental.optimizers.inverse_time_decay(initial_lr_x, decay_steps, decay_factor, staircase=True)
lr_y = jax.experimental.optimizers.inverse_time_decay(initial_lr_y, decay_steps, decay_factor, staircase=True)
# blocks = [int(b) for b in [block0, block1, block2, block3] if b > 0]

tb = mila_tools.deploy(host=HOST, sweep_yaml=sweep_yaml, extra_slurm_headers="""
#SBATCH --mem=24GB
""")

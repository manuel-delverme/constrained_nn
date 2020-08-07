import datetime
import os
import sys

import jax.experimental.optimizers
import sklearn.datasets
import tensorboardX

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

# dataset = sklearn.datasets.fetch_openml('mnist_784')
dataset = sklearn.datasets.load_iris()

# lr = jax.experimental.optimizers.constant(1e-3)
lr = jax.experimental.optimizers.inverse_time_decay(5e-3, 5000, 0.3, staircase=True)

num_experiments = 1
optimization_iters = 1000
optimization_subiters = 1000
num_hidden = 32
batch_size = 64
adam_betas = (0.3, 0.99)
adam_eps = 1e-8
use_sgd = False

comment = None
if not DEBUG:
    try:
        import tkinter.simpledialog

        root = tkinter.Tk()
        comment = tkinter.simpledialog.askstring("experiment_id", "experiment_id")
        root.destroy()
    except Exception as e:
        pass

if comment is None:
    comment = ""

current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
logdir = os.path.join('runs', current_time + '_' + comment)
tb = tensorboardX.SummaryWriter(logdir=logdir)

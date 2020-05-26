import datetime
import os

import sklearn.datasets
import tensorboardX

# dataset = sklearn.datasets.fetch_openml('mnist_784')
dataset = sklearn.datasets.load_iris()
num_hidden = 32
batch_size = 64
use_sgd = 0

comment = None
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

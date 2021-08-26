import sys

RUN_SWEEP = 1
REMOTE = 1

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
dataset_path = "../data" if DEBUG else "/network/datasets/{}.var/{}_torchvision"

experiment = ["sgd", "target-prop"][1]
constraint_satisfaction = ["penalty", "descent-ascent", "extra-gradient"][2]
dataset = ["mnist", "cifar10", "imagenet"][2]
distributional = True

# # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0
initial_forward = True
random_seed = 1337
num_layers = 0
warmup_lr = 0.009185

warmup_epochs = 0  # 1 if DEBUG else 0
num_epochs = 431
use_cuda = True  # not DEBUG

device = "cuda"
tabular_margin = 2.538115624834669
lambda_ = None

model_log = "all"
model_log_freq = 10000

workers = 0 if DEBUG else 4
epochs = 90
start_epoch = 0
batch_size = 8 if DEBUG else 256
resume = False
device = "cuda"

# Cifar10 params
distributional_margin = 0.2470519487851573
initial_lr_theta = 0.0001
initial_lr_x = 0.01
initial_lr_y = 9.356607499463217e-07
num_samples = 32

evaluate = False

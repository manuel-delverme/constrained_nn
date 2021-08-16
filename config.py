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

################################################################
# END OF PARAMETERS
################################################################
experiment_buddy.register_defaults(locals())

if distributional:
    assert experiment == "target-prop"

################################################################
# Derivative parameters
################################################################
device = torch.device("cuda" if use_cuda else "cpu")

if dataset == "mnist":
    if constraint_satisfaction == "extra-gradient":
        if distributional:
            # WARNING: these are not the best hyperparameters
            num_samples = 32
            distributional_margin = 0.3967
            initial_lr_theta = 0.0003638
            initial_lr_x = 0.05649
            initial_lr_y = 3.725e-07
        else:
            tabular_margin = 0.4373272842992752
            initial_lr_theta = 0.0008636215301536897
            initial_lr_x = 0.12499896839056827
            initial_lr_y = 7.270811457366213e-06
    elif constraint_satisfaction == "penalty":
        tabular_margin = 0.1017
        initial_lr_theta = 0.0003638
        initial_lr_x = 0.05649
        initial_lr_y = 3.725e-7
        lambda_ = 0.06788
        # 1e-2  # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0
    elif constraint_satisfaction == "descent-ascent":
        tabular_margin = 0.9658136136534436
        initial_lr_theta = 0.003314
        initial_lr_x = 0.04527
        initial_lr_y = 0.0001389
elif dataset == "cifar10":
    if constraint_satisfaction == "extra-gradient":
        if distributional:
            distributional_margin = 0.2470519487851573
            initial_lr_theta = 0.004169182899797638
            initial_lr_x = 0.25530572068931
            initial_lr_y = 9.356607499463217e-07
            num_samples = 32
        else:
            tabular_margin = 0.06640108363973078
            initial_lr_theta = 0.003175211334723672
            initial_lr_x = 0.03977922031861909
            initial_lr_y = 2.311834855494428e-06
    elif constraint_satisfaction == "penalty":
        tabular_margin = 0.10063086881740957
        initial_lr_theta = 4.6397470184556474e-05
        initial_lr_x = 0.27691927629931706
        initial_lr_y = 0.004094357077137722
        lambda_ = 0.06788
    elif constraint_satisfaction == "descent-ascent":
        tabular_margin = 0.13308695791662822
        initial_lr_theta = 0.00029136889726434325
        initial_lr_x = 0.25009935678225476
        initial_lr_y = 0.00015235056347032218

tb = experiment_buddy.deploy(
    host="mila" if REMOTE else "",
    sweep_yaml="test_suite.yaml" if RUN_SWEEP else False,
    extra_slurm_headers="""
    """,
    proc_num=20 if RUN_SWEEP else 1
)

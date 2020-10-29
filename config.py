import datetime
import getpass
import math
import os
import subprocess
import sys
import time
import types

import git
import jax.experimental.optimizers
import matplotlib.pyplot as plt
import tensorboardX
import wandb

sweep_yaml = "sweep_toy.yaml"
RUN_SWEEP = 0
CLUSTER = 1
PROJECT_NAME = "constrained_nn"

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

RANDOM_SEED = 1337

dataset = "iris"
num_hidden = 32
initial_lr_theta = .01
initial_lr_x = .01
initial_lr_y = .01
# 1e-2  # high lr_y make the lagrangian more responsive to sign changes -> less oscillation around 0

blocks = [2, ] * 25


def state_fn(x):
    # x = stax.softplus(x)
    # x = stax.relu(x)
    # x = stax.leaky_relu(x)
    # x = stax.softplus(x)
    return x


use_adam = False
grad_clip = False  # 0.5  # avoid leaky_grad explosions
adam1 = 0.9
adam2 = 0.99

batch_size = 150
weight_norm = False  # avoid unbound targets

num_epochs = 1000000
eval_every = math.ceil(num_epochs / 1000)

decay_steps = num_epochs // 4  # 500000
decay_factor = 1.0

################################################################
# END OF PARAMETERS
################################################################
config_params = locals().copy()

wandb_encoding = "^"
# overwrite CLI parameters
# fails on nested config object
for arg in sys.argv[1:]:
    assert arg[:2] == "--"
    k, v = arg[2:].split("=")
    k = k.lstrip(wandb_encoding)

    if "." in v:
        v = float(v)
    else:
        try:
            v = int(v)
        except ValueError:
            pass

    if k not in config_params.keys():
        raise ValueError(f"Trying to set {k}, but that's not one of {list(config_params.keys())}")
    locals()[k] = v

################################################################
# Derivative parameters
################################################################
# initial_lr_y = initial_lr_x * 10.
lr_theta = jax.experimental.optimizers.inverse_time_decay(initial_lr_theta, decay_steps, decay_factor, staircase=True)
lr_x = jax.experimental.optimizers.inverse_time_decay(initial_lr_x, decay_steps, decay_factor, staircase=True)
lr_y = jax.experimental.optimizers.inverse_time_decay(initial_lr_y, decay_steps, decay_factor, staircase=True)


# everything below should not be here but refactored away


class Wandb:
    _global_step = 0

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value

    def __init__(self, _experiment_id):
        print(f"wandb.init(project={PROJECT_NAME}, name={_experiment_id})")
        wandb.init(project=PROJECT_NAME, name=_experiment_id)

        def register_param(_k, _v, prefix=""):
            if _k.startswith("__") and _k.endswith("__"):
                return
            if _k == "_":
                return
            if isinstance(_v, (types.FunctionType, types.MethodType, types.ModuleType)):
                return

            if _k == "_extra_modules_":
                for module in _v:
                    for __k in dir(module):
                        __v = getattr(module, __k)
                        register_param(__k, __v, prefix=module.__name__.replace(".", "_"))
            else:
                key = prefix + wandb_encoding + _k
                # if the parameter was not set by a sweep
                if not key in wandb.config._items:
                    print(f"setting {key}={str(_v)}")
                    setattr(wandb.config, key, str(_v))
                else:
                    print(f"not setting {key} to {str(_v)}, str because its already {getattr(wandb.config, key)}, {type(getattr(wandb.config, key))}")

        params = config_params
        for k, v in params.items():
            register_param(k, v)

    def add_scalar(self, tag, scalar_value, global_step=None):
        wandb.log({tag: scalar_value}, step=global_step, commit=False)

    def add_figure(self, tag, figure, global_step=None, close=True):
        wandb.log({tag: figure}, step=global_step, commit=False)
        if close:
            plt.close(figure)

    def add_histogram(self, tag, values, global_step=None):
        wandb.log({tag: wandb.Histogram(values)}, step=global_step, commit=False)


git_repo = git.Repo(os.path.dirname(__file__))


def setup_tb(logdir):
    tb = tensorboardX.SummaryWriter(logdir=logdir)
    print("http://localhost:6006")
    return tb


def commit_and_sendjob(experiment_id):
    # 2) commits everything to git with the name as message (so i r later reproduce the same experiment)
    os.system(f"git add .")
    os.system(f"git commit -m '{experiment_id}'")
    # 3) pushes the changes to git
    os.system("git push")

    if RUN_SWEEP:
        cmd_list = f"/home/esac/research/fax/venv/bin/wandb sweep --name {experiment_id} -p {PROJECT_NAME} {sweep_yaml}".split(" ")
        wandb_stdout = subprocess.check_output(cmd_list, stderr=subprocess.STDOUT).decode("utf-8")
        print(wandb_stdout)
        sweep_id = wandb_stdout.split("/")[-1].strip()
        command = f"ssh mila /opt/slurm/bin/sbatch ./localenv_sweep.sh https://github.com/manuel-delverme/OptimalControlNeuralNet {sweep_id} {git_repo.commit().hexsha}"
        num_repeats = 1  # this should become > 1 for parallel sweeps
    else:
        main = sys.argv[0].split(os.getcwd())[-1].lstrip("/")
        command = f"ssh mila bash -l ./run_experiment.sh https://github.com/manuel-delverme/OptimalControlNeuralNet {main} {git_repo.commit().hexsha}"
        num_repeats = 1  # this should become > 1 for parallel sweeps

    # command = f"ssh mila ./run_experiment.sh {next(git_repo.remote().urls)} {main} {git_repo.commit().hexsha}"
    print(command)
    for proc_num in range(num_repeats):
        if proc_num > 0:
            time.sleep(1)
            raise NotImplemented
        if proc_num > 1:
            priority = "long"
            raise NotImplemented("localenv_sweep.sh does not handle this yet")

        os.system(command)
    with open("ssh.log", 'w') as fout:
        fout.write(command)
    # 4) logs on the server and pulls the latest version
    # 5) runs the experiment
    # 7) writes me on slack/telegram/email a link for the tensorboard


if (not CLUSTER) and RUN_SWEEP:
    raise ValueError("?!?!?")

if getpass.getuser() == 'delvermm':
    print("using wandb")
    experiment_id = f"{git_repo.head.commit.message.strip()}"
    dtm = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    tb = Wandb(f"{experiment_id}_{dtm}")
else:
    experiment_id = None
    if not DEBUG:
        try:
            import tkinter.simpledialog

            root = tkinter.Tk()
            if CLUSTER:
                experiment_id = tkinter.simpledialog.askstring("[CLUSTER]", "experiment_id")
            else:
                experiment_id = tkinter.simpledialog.askstring("[LOCAL]", "experiment_id")
            experiment_id.replace(" ", "_")
            root.destroy()
        except Exception as e:
            pass

    if CLUSTER:
        experiment_id = "[CLUSTER]" + experiment_id
        commit_and_sendjob(experiment_id)
        sys.exit()
    else:
        dtm = datetime.datetime.now().strftime("%b%d_%H-%M-%S") + ".pt/"
        # experiment_id = f"{git_repo.head.commit.message.strip()}"
        experiment_id = experiment_id or f"DEBUG_RUN"
        # tb = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.join(C.TENSORBOARD, experiment_id, dtm))
        tb = setup_tb(logdir=os.path.join("tensorboard/", experiment_id, dtm))

print(f"experiment_id: {experiment_id}", dtm)
tb.global_step = 0

if any((DEBUG,)):
    print(r'''
                                    _.---"'"""""'`--.._
                             _,.-'                   `-._
                         _,."                            -.
                     .-""   ___...---------.._             `.
                     `---'""                  `-.            `.
                                                 `.            \
                                                   `.           \
                                                     \           \
                                                      .           \
                                                      |            .
                                                      |            |
                                _________             |            |
                          _,.-'"         `"'-.._      :            |
                      _,-'                      `-._.'             |
                   _.'                              `.             '
        _.-.    _,+......__                           `.          .
      .'    `-"'           `"-.,-""--._                 \        /
     /    ,'                  |    __  \                 \      /
    `   ..                       +"  )  \                 \    /
     `.'  \          ,-"`-..    |       |                  \  /
      / " |        .'       \   '.    _.'                   .'
     |,.."--"""--..|    "    |    `""`.                     |
   ,"               `-._     |        |                     |
 .'                     `-._+         |                     |
/                           `.                        /     |
|    `     '                  |                      /      |
`-.....--.__                  |              |      /       |
   `./ "| / `-.........--.-   '              |    ,'        '
     /| ||        `.'  ,'   .'               |_,-+         /
    / ' '.`.        _,'   ,'     `.          |   '   _,.. /
   /   `.  `"'"'""'"   _,^--------"`.        |    `.'_  _/
  /... _.`:.________,.'              `._,.-..|        "'
 `.__.'                                 `._  /
                                           "'
  ''')

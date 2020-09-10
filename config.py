import datetime
import getpass
import os
import subprocess
import sys
import types

import git
import jax.experimental.optimizers
import matplotlib.pyplot as plt
import sklearn.datasets
import tensorboardX
import wandb

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()
RUN_SWEEP = True
PROJECT_NAME = "constrained_nn"

if DEBUG:
    print("USING IRIS DATASET")
    dataset = lambda: sklearn.datasets.load_iris()
else:
    dataset = lambda: sklearn.datasets.fetch_openml('mnist_784')

RANDOM_SEED = 1337

# lr = jax.experimental.optimizers.constant(1e-3)
optimization_iters = int(1e6)

initial_lr = 1e-4
lr = jax.experimental.optimizers.inverse_time_decay(initial_lr, 1000, 0.3, staircase=True)

# optimization_subiters = 1000
num_hidden = 128
eval_every = 100
batch_size = 64

adam1 = 0.9
adam2 = 0.99
adam_betas = adam1, adam2
adam_eps = 1e-8
use_sgd = False

constrained = 1

################################################################
# END OF PARAMETERS
################################################################
config_params = locals().copy()

# everything below should not be here, what are you going to do? call the police?.

params = config_params
for k, v in params.items():
    pass


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
                key = prefix + "_" + _k
                # if the parameter was not set by a sweep
                if not key in wandb.config._items:
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
    # 2) commits everything to git with the name as message (so i can later reproduce the same experiment)
    os.system(f"git add .")
    os.system(f"git commit -m '[CLUSTER] {experiment_id}'")
    # 3) pushes the changes to git
    os.system("git push")

    if RUN_SWEEP:
        wandb_stdout = subprocess.check_output(f"wandb sweep --name {experiment_id} -p {PROJECT_NAME} sweep.yaml".split(" "), stderr=subprocess.STDOUT).decode("utf-8")
        print(wandb_stdout)
        sweep_id = wandb_stdout.split("/")[-1].strip()
        command = f"ssh mila /opt/slurm/bin/sbatch ./localenv_sweep.sh https://github.com/manuel-delverme/OptimalControlNeuralNet {sweep_id} {git_repo.commit().hexsha}"
    else:
        main = sys.argv[0].split(os.getcwd())[-1].lstrip("/")
        command = f"ssh mila bash -l ./run_experiment.sh https://github.com/manuel-delverme/OptimalControlNeuralNet {main} {git_repo.commit().hexsha}"

    # command = f"ssh mila ./run_experiment.sh {next(git_repo.remote().urls)} {main} {git_repo.commit().hexsha}"
    print(command)
    os.system(command)
    with open("ssh.log", 'w') as fout:
        fout.write(command)
    # 4) logs on the server and pulls the latest version
    # 5) runs the experiment
    # 7) writes me on slack/telegram/email a link for the tensorboard


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
            experiment_id = tkinter.simpledialog.askstring("experiment_id", "experiment_id")
            root.destroy()
        except Exception as e:
            pass

    if experiment_id is None:
        dtm = datetime.datetime.now().strftime("%b%d_%H-%M-%S") + ".pt/"
        # experiment_id = f"{git_repo.head.commit.message.strip()}"
        experiment_id = f"DEBUG_RUN"
        # tb = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.join(C.TENSORBOARD, experiment_id, dtm))
        tb = setup_tb(logdir=os.path.join("tensorboard/", experiment_id, dtm))
    else:
        commit_and_sendjob(experiment_id)
        sys.exit()

print(f"experiment_id: {experiment_id}")
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

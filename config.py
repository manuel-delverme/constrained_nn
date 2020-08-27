import datetime
import getpass
import os
import sys
import types

import git
import jax.experimental.optimizers
import matplotlib.pyplot as plt
import sklearn.datasets
import tensorboardX
import wandb

DEBUG = '_pydev_bundle.pydev_log' in sys.modules.keys()

if DEBUG:
    print("USING IRIS DATASET")
    dataset = lambda: sklearn.datasets.load_iris()
else:
    dataset = lambda: sklearn.datasets.fetch_openml('mnist_784')

RANDOM_SEED = 1337

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

constrained = 1

max_episode_steps = 10
n_eval_episodes = 100
max_samples = paper_evaluation_max_samples = int(1e6)
paper_evaluation_n_eval_episodes = 100

critic_math = True
actor_math = True

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
        wandb.init(project="constrained_nn", name=_experiment_id)

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
                setattr(wandb.config, prefix + "_" + _k, str(_v))

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


def commit_and_sendjob():
    global experiment_id
    # 2) commits everything to git with the name as message (so i can later reproduce the same experiment)
    print("git add")
    os.system(f"git add .")
    print("git commit")
    os.system(f"git commit -m '[CLUSTER] {experiment_id}'")
    # 3) pushes the changes to git
    print("git push")
    os.system("git push")
    main = sys.argv[0].split(os.getcwd())[-1].lstrip("/")
    # command = f"ssh mila ./run_experiment.sh {next(git_repo.remote().urls)} {main} {git_repo.commit().hexsha}"
    command = f"ssh mila bash -l ./run_experiment.sh https://github.com/manuel-delverme/OptimalControlNeuralNet {main} {git_repo.commit().hexsha}"
    print(command)
    os.system(command)
    with open("ssh.log", 'a') as fout:
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
        commit_and_sendjob()
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

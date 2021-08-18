import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import wandb


def disk_cache(f):
    def wrapper(*args, **kwargs):
        fid = f.__name__
        cache_file = "cache/{}".format(fid)
        if args:
            if not os.path.exists(cache_file):
                os.makedirs(cache_file)
            fid = fid + "/" + "::".join(str(arg) for arg in args).replace("/", "_")
            cache_file = "cache/{}".format(fid)
        cache_file += ".pkl"
        try:
            with open(cache_file, "rb") as fin:
                cached_return = pickle.load(fin)
        except FileNotFoundError:
            cached_return = f(*args, **kwargs)
            with open(cache_file, "wb") as file_out:
                pickle.dump(cached_return, file_out)
        return cached_return

    return wrapper


def main():
    min_timestamp = None
    mnist_sweeps = [
        "ijz3s64d",
        "6cfcvtam",
        "933xomle",
        "ybu8nps8",
    ]
    baselines = {
        "difference_target_propagation/runs/7rfgdxkq": "DTP",
        "online-alt-min/runs/73jzusur": "OAM"
    }
    sweep_pretty_names = {
        "ijz3s64d": "Gaussian States",
        "6cfcvtam": "Regularization",
        "933xomle": "GDA",
        "ybu8nps8": "Tabular States",
        **baselines
    }
    keys_to_plot = [
        'train/loss',
        'test/accuracy',
        'test/loss',
        'h0/train/abs_mean_defect',
    ]
    pretty_keys = {
        'train/loss': "Train Loss",
        'test/accuracy': "Test Error",
        'test/loss': "Test Loss",
        'h0/train/abs_mean_defect': "Mean Defect Magnitude",
    }

    sweeps_data = get_data(keys_to_plot, mnist_sweeps, baselines)

    # for k in data:
    #     data[k] = np.array(data[k])

    mnist_sweeps = [*mnist_sweeps, *baselines.keys()]
    for k in keys_to_plot:
        fig, ax = plt.subplots(1)
        title = pretty_keys[k]
        ax.set_title(title)
        min_time = float('inf')
        ax.set_xlabel('mini-batches')
        for sweep_id, data in zip(mnist_sweeps, sweeps_data):
            if not data[k]:
                continue
            time = [d.to_numpy() for d in data['_step']]
            values = [row.to_numpy() for row in data[k]]
            min_time = min(min_time, min(len(r) for r in values))
            print(min_time)
            min_timestamp = time[0][min_time - 1]

            values = [row[:min_time] for row in values]
            values = np.stack(values)
            time = [row[:min_time] for row in time]
            time = np.stack(time)[0]

            # N steps length arrays empirical means and standard deviations of both
            # populations over time
            mu1 = values.mean(axis=0)
            sigma1 = values.std(axis=0)

            # 'train/loss': "Train Loss",
            # 'test/accuracy': "Test Accuracy",
            # 'test/loss': "Test Loss",
            # 'h0/train/multipliers_abs_mean': "Mean Multipliers Magnitude",
            # 'h0/train/abs_mean_defect': "Mean Defect Magnitude",
            if title == "Train Loss" or title == "Mean Defect Magnitude":  # or title ==:
                line, = ax.semilogy(time, mu1, lw=2, label=sweep_pretty_names[sweep_id], nonpositive="clip")
            elif title == "Test Error":  # or title == "Test Loss" or title ==:
                mu1 = 1 - mu1
                line, = ax.semilogy(time, mu1, lw=1, label=sweep_pretty_names[sweep_id])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
                # ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
            else:
                line, = ax.plot(time, mu1, lw=1, label=sweep_pretty_names[sweep_id])
            ax.fill_between(time, mu1 + sigma1, mu1 - sigma1, facecolor=line._color, alpha=0.3)  # noqa

        ax.legend(loc='upper right')
        ax.set_xlim(0, min_timestamp)
        # plt.show()
        plt.savefig(f"images/{title}.png", dpi=300)


@disk_cache
def get_data(keys_to_plot, mnist_sweeps, baselines):
    api = wandb.Api()
    data = []

    sweeps_runs = [api.sweep(f"delvermm/constrained_nn/sweeps/{sweep_id}").runs for sweep_id in mnist_sweeps]
    for baseline in baselines:
        sweeps_runs.append([api.run(f"delvermm/{baseline}", ), ])

    for sweep_runs in sweeps_runs:
        data_sweep = {k: list() for k in keys_to_plot + ["_step", ]}
        for run in sweep_runs:
            # run.summary are the output key/values like accuracy.
            # We call ._json_dict to omit large files
            history = run.history(keys=keys_to_plot, pandas=True, samples=1000)
            if len(history) == 0:
                keys_to_plot.remove('h0/train/abs_mean_defect')

                history = run.history(keys=keys_to_plot, pandas=True, samples=1000)
                keys_to_plot.append('h0/train/abs_mean_defect')

            for k in keys_to_plot + ["_step", ]:
                if k in history:
                    v = history[k]  # .to_numpy()
                    data_sweep[k].append(v)  # list(v))
        data.append(data_sweep)
    return data


main()

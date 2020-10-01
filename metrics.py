from jax import numpy as np

import config
import utils
from utils import train_accuracy


def update_metrics(_batches, equality_constraints, full_rollout_loss, loss_function, model, params, outer_iter, train_x, train_y):
    params, multipliers = params
    # _train_x, _train_y, _indices = next(batches)
    # metrics_time = time.time()
    # fullbatch = train_x, train_y, np.arange(train_x.shape[0])
    loss, task = loss_function(params)
    h, _task = equality_constraints(params, task)
    full_loss = full_rollout_loss(params.theta, task)

    metrics = [
                  ("train/train_accuracy", train_accuracy(train_x, train_y, model, params.theta)),
                  ("train/full_rollout_loss", full_loss),
                  ("train/loss", loss),
                  ("train/lr", config.lr(outer_iter)),
              ] + [
                  (f"constraints/{idx}_multipliers", np.linalg.norm(mi, 1)) for idx, mi in enumerate(multipliers)
              ] + [
                  (f"constraints/{idx}_defects", np.linalg.norm(hi, 1)) for idx, hi in enumerate(h)
              ] + [
                  (f"params/{idx}_x", np.linalg.norm(xi, 1)) for idx, xi in enumerate(params.x)
              ] + [
                  (f"params/{idx}_theta", (np.linalg.norm(p[0], 1) + np.linalg.norm(p[1], 1)) / 2) for idx, (p, _) in enumerate(params.theta)
              ] + [
                  (f"train/{t}_step_sampled_accuracy", utils.n_step_accuracy(*next(_batches), model, params, t)) for t in range(1, len(params.theta))
              ]
    # for idx, mi in enumerate(multipliers[:10]):
    #     for jdx, mij in enumerate(mi):
    #         metrics.append((f"AA/{idx}_multi_{jdx}", mij))

    # for idx, hi in enumerate(h[:10]):
    #     for jdx, hij in enumerate(hi):
    #         metrics.append((f"AA/{idx}_defect_{jdx}", hij))

    # for idx, xi in enumerate(params.x[:10]):
    #     for jdx, xij in enumerate(xi):
    #         metrics.append((f"AA/{idx}_x_{jdx}", xij))

    for tag, value in metrics:
        config.tb.add_scalar(tag, float(value), outer_iter)
    if outer_iter == 0:
        print(f"constraints/defects", )
        print([np.linalg.norm(hi, 1) for idx, hi in enumerate(h)])

import jax
from fax import math
from jax import numpy as np

import config
import utils
from utils import train_accuracy, time_march


def update_metrics(lagrangian, equality_constraints, full_rollout_loss, loss_function, model, params_lagr, outer_iter, train_x, train_y):
    params, multipliers = params_lagr
    loss = loss_function(params)
    lagr_val = lagrangian(*params_lagr)
    h = equality_constraints(params)
    # a0 = model[0](params.theta[0], train_x)
    a = time_march(train_x, model, params.theta)  # These are wrong, use 1 step forward prop

    full_loss = full_rollout_loss(params.theta)
    rhs = []
    for mi, hi in zip(multipliers, h):
        rhs.append(math.pytree_dot(mi, hi))
        # regul += np.linalg.norm(hi, 2)

    metrics = [
                  ("train/train_accuracy", train_accuracy(train_x, train_y, model, params.theta)),
                  ("train/full_rollout_loss", full_loss),
                  ("train/lagrangian", lagr_val),
                  ("train/loss", loss),
                  ("train/lr_x", config.lr_x(outer_iter)),
                  ("train/lr_y", config.lr_y(outer_iter)),
              ] + [
                  (f"train/{idx}_rhs", rhsi) for idx, rhsi in enumerate(rhs)
              ] + [
                  (f"constraints/{idx}_multipliers", np.linalg.norm(mi, 1)) for idx, mi in enumerate(multipliers)
              ] + [
                  (f"constraints/{idx}_defects", np.linalg.norm(hi, 1)) for idx, hi in enumerate(h)
              ] + [
                  (f"params/{idx}_a", np.linalg.norm(ai, 1)) for idx, ai in enumerate(a)
              ] + [
                  (f"params/{idx}_th(x)", np.linalg.norm(config.state_fn(xi), 1)) for idx, xi in enumerate(params.x)
              ] + [
                  (f"params/{idx}_theta", (np.linalg.norm(p[0], 1) + np.linalg.norm(p[1], 1)) / 2) for idx, (p, _) in enumerate(params.theta[:-1])
              ] + [
                  (f"train/{t}_step_sampled_accuracy", utils.n_step_accuracy(train_x, train_y, model, params, t)) for t in range(1, len(params.theta))
              ]
    if False:
        (g_p, g_multi) = jax.grad(lagrangian, (0, 1))(*params_lagr)
        for idx, g_theta_i in enumerate(g_p.x):
            for jdx, g_theta_ij in enumerate(g_theta_i):
                metrics.append((f"AA/{idx}_g_x_{jdx}", g_theta_ij))

        for idx, g_theta_i in enumerate(g_p.theta):
            for jdx, g_theta_ij in enumerate(g_theta_i[0]):
                metrics.append((f"AA/{idx}_g_theta_|w|_{jdx}", np.linalg.norm(g_theta_ij, 2)))

        for idx, mi in enumerate(multipliers[:1]):
            for jdx, mij in enumerate(mi):
                metrics.append((f"AA/{idx}_multi_{jdx}", mij))

        for idx, hi in enumerate(h[:1]):
            for jdx, hij in enumerate(hi):
                metrics.append((f"AA/layer_{idx}_defect_{jdx}", hij))

        for idx, xi in enumerate(params.x[:1]):
            for jdx, xij in enumerate(config.state_fn(xi)):
                metrics.append((f"AA/{idx}_x_tar_{jdx}", xij))

        for idx, ai in enumerate(a[:-1]):
            for jdx, aij in enumerate(ai):
                if aij.shape[0] != 1:
                    raise ValueError("config.num_hidden = 1")
                metrics.append((f"AA/{idx}_x_hat_{jdx}", float(aij)))

        # for idx, pi in enumerate(params.theta[0]):
        #     for jdx, pij in enumerate(pi):
        #         metrics.append((f"AA/{idx}_p_{jdx}", pij))

        # for idx, pi in enumerate(params.theta[0][0][0]):
        #     for jdx, pij in enumerate(pi):
        #         metrics.append((f"AA/{idx}_p0_{jdx}", pij))

        # for idx, pi in enumerate(params.theta[0][0][1]):
        #     metrics.append((f"AA/{idx}_b0", pi))

        # for idx, pi in enumerate(params.theta[1][0][0]):
        #     for jdx, pij in enumerate(pi):
        #         metrics.append((f"AA/{idx}_p1_{jdx}", pij))

        # for idx, pi in enumerate(params.theta[1][0][1]):
        #     metrics.append((f"AA/{idx}_b1", pi))

    for tag, value in metrics:
        config.tb.add_scalar(tag, float(value), outer_iter)
    if outer_iter == 0:
        print(f"constraints/defects", )
        print([np.linalg.norm(hi, 1) for idx, hi in enumerate(h)])

import jax
from fax import math
from jax import numpy as np

import config
import utils
from utils import train_accuracy

old_metrics = None


def update_metrics(lagrangian, equality_constraints, full_rollout_loss, loss_function, model, parameters: utils.LagrangianParameters, outer_iter, full_batch):
    global old_metrics
    constr_params, multipliers = parameters
    loss, batch = loss_function(constr_params)
    lagr_val = lagrangian(*parameters)
    h, constraints_batch = equality_constraints(constr_params, batch)
    # a = time_march(full_batch.x, model, constr_params.theta)  # These are wrong, use 1 step forward prop
    a = utils.one_step(full_batch.x, constr_params.x, model, constr_params.theta)
    full_loss = full_rollout_loss(constr_params.theta, batch)
    rhs = []
    for mi, hi in zip(multipliers, h):  # TODO: sample these
        rhs.append(math.pytree_dot(mi[constraints_batch.indices], hi))

    n_step_acc = [utils.n_step_accuracy(full_batch.x, full_batch.y, model, constr_params, t) for t in range(1, len(constr_params.theta))]
    meta_obj = np.cumproduct(np.array(n_step_acc))[-1]
    if old_metrics is not None:
        meta_obj = dict(old_metrics)["train/meta_obj"] * 0.9 + meta_obj * 0.1

    metrics = [
                  ("train/train_accuracy", train_accuracy(full_batch.x, full_batch.y, model, constr_params.theta)),
                  ("train/full_rollout_loss", full_loss),
                  ("train/lagrangian", lagr_val),
                  ("train/loss", loss),
                  ("train/lr_x", config.lr_x(outer_iter)),
                  ("train/lr_y", config.lr_y(outer_iter)),
              ] + [
                  (f"train/rhs_{idx}", rhsi) for idx, rhsi in enumerate(rhs)
              ] + [
                  (f"constraints/multipliers_{idx}", np.linalg.norm(mi, 1)) for idx, mi in enumerate(multipliers)
              ] + [
                  (f"constraints/defects_{idx}", np.mean(np.linalg.norm(hi, 1))) for idx, hi in enumerate(h)
              ] + [
                  (f"params/f(x,theta)_max_{idx}", np.max(ai)) for idx, ai in enumerate(a[:-1])
              ] + [
                  (f"params/f(x,theta)_min_{idx}", np.min(ai)) for idx, ai in enumerate(a[:-1])
              ] + [
                  (f"params/f(x,theta)_sum_{idx}", np.mean(np.sum(ai, 1))) for idx, ai in enumerate(a[:-1])
                  # ] + [
                  #     (f"params/a_max_{idx}", np.max(config.state_fn(xi))) for idx, xi in enumerate(constr_params.x)
                  # ] + [
                  #     (f"params/a_sum_{idx}", np.mean(np.sum(config.state_fn(xi), 1))) for idx, xi in enumerate(constr_params.x)
                  # ] + [
                  #     (f"params/theta_{idx}", (np.linalg.norm(p[0], 1) + np.linalg.norm(p[1], 1)) / 2) for idx, (p, _) in enumerate(constr_params.theta[:-1])
              ] + [
                  (f"train/step_accuracy_{t + 1}", t_acc) for t, t_acc in enumerate(n_step_acc)
              ] + [
                  (f"train/meta_obj", meta_obj)
              ]
    if False:
        (g_p, g_multi) = jax.grad(lagrangian, (0, 1))(*parameters)
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

        for idx, xi in enumerate(constr_params.x[:1]):
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

    found_nan = False
    for tag, value in metrics:
        config.tb.add_scalar(tag, float(value), outer_iter)
        if value != value or abs(value) > 1e10:
            print(dict(old_metrics)[tag], tag, value)
            found_nan = True
    if found_nan:
        raise ValueError("Found NaN")

    if outer_iter == 0:
        print(f"constraints/defects", )
        print([np.linalg.norm(hi, 1) for idx, hi in enumerate(h)])

    old_metrics = metrics

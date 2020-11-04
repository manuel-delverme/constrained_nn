import fax.utils as fax_utils
from fax import math
from jax import numpy as np

import config
import utils
from utils import train_acc

old_metrics = None


def update_metrics(lagrangian, make_losses, model, p: fax_utils.LagrangianParameters, outer_iter, full_batch):
    global old_metrics

    full_rollout_loss, one_step_loss, equality_constraints = make_losses(model)
    h = equality_constraints(p, full_batch)
    a = utils.one_step(full_batch.x, p.constr_params.x, model, p.constr_params.theta)
    rhs = []
    for mi, hi in zip(p.multipliers, h):
        rhs.append(math.pytree_dot(mi, hi))

    n_step_loss, n_step_acc = zip(*[utils.n_step_acc(full_batch.x, full_batch.y, model, p.constr_params, t + 1) for t, _ in enumerate(p.constr_params.theta)])
    # TODO: track n_step_losses
    meta_obj = np.mean(np.array(n_step_acc))
    if old_metrics is not None:
        meta_obj = dict(old_metrics)["train/meta_obj"] * 0.9 + meta_obj * 0.1

    full_batch_loss, full_batch_train_acc = train_acc(full_batch.x, full_batch.y, model, p.constr_params.theta)

    metrics = [
                  ("train/full_rollout_loss", full_batch_loss),
                  ("train/lagrangian", lagrangian(p, full_batch)),
                  ("train/full_batch_train_accuracy", full_batch_train_acc),
                  # ("train/full_batch_loss", full_batch_loss),
                  ("train/lr_x", config.lr_x(outer_iter)),
                  ("train/lr_y", config.lr_y(outer_iter)),
              ] + [
                  (f"train/rhs_{idx}", rhsi) for idx, rhsi in enumerate(rhs)
              ] + [
                  (f"constraints/multipliers_{idx}", np.linalg.norm(mi, 1)) for idx, mi in enumerate(p.multipliers)
              ] + [
                  (f"constraints/defects_{idx}", np.mean(np.linalg.norm(hi, 1))) for idx, hi in enumerate(h)
              ] + [
                  (f"params/f(x,theta)_max_{idx}", np.max(ai)) for idx, ai in enumerate(a[:-1])
              ] + [
                  (f"params/f(x,theta)_min_{idx}", np.min(ai)) for idx, ai in enumerate(a[:-1])
                  # ] + [
                  #     (f"params/f(x,theta)_sum_{t}", np.mean(np.sum(ai, 1))) for t, ai in enumerate(a[:-1])
              ] + [
                  (f"params/x_min_{idx}", np.min(config.state_fn(xi))) for idx, xi in enumerate(p.constr_params.x)
              ] + [
                  (f"params/x_max_{idx}", np.max(config.state_fn(xi))) for idx, xi in enumerate(p.constr_params.x)
                  # ] + [
                  #     (f"params/a_sum_{t}", np.mean(np.sum(config.state_fn(xi), 1))) for t, xi in enumerate(constr_params.x)
                  # ] + [
                  #     (f"params/theta_{t}", (np.linalg.norm(p[0], 1) + np.linalg.norm(p[1], 1)) / 2) for t, (p, _) in enumerate(constr_params.theta[:-1])
              ] + [
                  (f"train/step_accuracy_{t + 1}", t_acc) for t, t_acc in enumerate(n_step_acc)
              ] + [
                  (f"train/step_loss_{t + 1}", t_acc) for t, t_acc in enumerate(n_step_loss)
              ] + [
                  (f"train/meta_obj", meta_obj)
              ]

    if config.num_hidden == 1:
        # (g_p, g_multi) = jax.grad(lagrangian, (0, 1))(*parameters)
        # for t, g_theta_i in enumerate(g_p.x):
        #     for jdx, g_theta_ij in enumerate(g_theta_i):
        #         metrics.append((f"AA/grad_x_{t}_{jdx}", g_theta_ij))

        # for t, g_theta_i in enumerate(g_p.theta):
        #     for jdx, g_theta_ij in enumerate(g_theta_i[0]):
        #         metrics.append((f"AA/g_theta_|w|_{t}_{jdx}", np.linalg.norm(g_theta_ij, 2)))

        for t, mi in enumerate(p.multipliers):
            for jdx, mij in enumerate(mi):
                metrics.append((f"AA/block_{t}_multi_{jdx}", mij))

        for t, hi in enumerate(h):
            for jdx, hij in enumerate(hi):
                metrics.append((f"AA/block_{t}_defect_{jdx}", hij))

        for t, xi in enumerate(p.constr_params.x):
            for jdx, xij in enumerate(config.state_fn(xi)):
                metrics.append((f"AA/block_{t}_x_{jdx}", xij))

        for t, ai in enumerate(a[:-1]):
            for jdx, aij in enumerate(ai):
                metrics.append((f"AA/block_{t}_x_hat_{jdx}", float(aij)))

        for t, pi in enumerate(p.constr_params.theta):
            param, bias = pi[0]
            for jdx, bj in enumerate(bias):
                metrics.append((f"AA/block_{t}_fc_bias_{jdx}", bj))

            if param.shape[0] == 1:
                param = param.copy().transpose()

            for jdx, pij in enumerate(param):
                metrics.append((f"AA/block_{t}_fc_w_{jdx}_0", pij))

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
        print([np.max(hi, 1) for idx, hi in enumerate(h)])

    old_metrics = metrics

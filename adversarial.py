# train_x, train_y, model, theta, x, y
import time
from typing import List

import fax
import fax.competitive.extragradient
import fax.constrained
import fax.math
import jax
import jax.experimental.optimizers
import jax.lax
import jax.numpy as np
import jax.ops
import jax.tree_util
import numpy.random as npr
import tqdm

import config
import datasets
from network import make_block_net
from utils import ConstrainedParameters, TaskParameters, make_n_step_loss, n_step_accuracy, forward_prop, time_march, train_accuracy


def main():
    batch_gen, model, params, train_x, train_y = initialize()

    def full_rollout_loss(theta: List[np.ndarray], batch):
        train_x, batch_train_y, _indices = batch
        pred_y = forward_prop(train_x, model, theta)
        return np.linalg.norm(pred_y - batch_train_y, 2)

    onestep = make_n_step_loss(1, full_rollout_loss, batch_gen)

    def equality_constraints(params, task):
        theta, x = params
        task_x, _, task_indices = task

        # Layer 1 -> 2
        defects = [x[0][task_indices, :] - model[0](theta[0], task_x), ]

        # Layer 2 onward
        for t in range(len(x) - 1):
            block_x = x[t][task_indices, :]
            block_y = x[t + 1][task_indices, :]
            block_y_hat = model[t + 1](theta[t + 1], block_x)
            defects.append(np.square(block_y - block_y_hat))
            # defects.append(0.)
        return tuple(defects), task_indices

    # initial_values = params
    optimizer_init, optimizer_update, optimizer_get_params = fax.competitive.extragradient.adam_extragradient_optimizer(betas=(config.adam1, config.adam2), step_size=config.lr)
    opt_state = optimizer_init(params)

    def lagrangian(params):
        obj, task = onestep(params)
        h, _task = equality_constraints(params, task)
        regul = 0.
        for hi in h:
            regul += np.linalg.norm(hi, 2)
        return obj + regul

    @jax.jit
    def update(i, opt_state):
        grad_fn = jax.grad(lagrangian, (0, 1))
        return optimizer_update(i, grad_fn, opt_state)

    print("optimize()")
    update_time = time.time()

    for iter_num in tqdm.trange(config.num_epochs):
        opt_state = update(iter_num, opt_state)

        update_time = time.time() - update_time
        params = optimizer_get_params(opt_state)
        if iter_num % config.eval_every == 0:
            update_metrics(batch_gen, equality_constraints, full_rollout_loss, model, params, iter_num, update_time, train_x, train_y)
            update_time = time.time()

    trained_params = optimizer_get_params(opt_state)
    return trained_params


def update_metrics(_batches, equality_constraints, full_rollout_loss, model, params, outer_iter, update_time, train_x, train_y):
    params, multipliers = params
    # _train_x, _train_y, _indices = next(batches)
    metrics_time = time.time()
    fullbatch = train_x, train_y, np.arange(train_x.shape[0])
    h, _task = equality_constraints(params, fullbatch)

    def b():
        while True:
            yield fullbatch

    batches = b()

    metrics = [
                  ("train/train_accuracy", train_accuracy(train_x, train_y, model, params.theta)),
                  ("train/full_rollout_loss", full_rollout_loss(params.theta, next(batches))),
                  ("train/1step_loss", make_n_step_loss(1, full_rollout_loss, batches)(params)[0]),
                  # ("train/multipliers", np.linalg.norm(multipliers, 1)),
                  # ("train/update_time", time.time() - update_time),
              ] + [
                  (f"constraints/multipliers_{idx}", np.linalg.norm(mi, 2)) for idx, mi in enumerate(multipliers)
              ] + [
                  (f"constraints/defects_{idx}", np.linalg.norm(hi, 2)) for idx, hi in enumerate(h)
              ] + [
                  (f"train/{t}_step_accuracy", n_step_accuracy(*fullbatch, model, params, t)) for t in range(1, len(params.theta))
                  # ] + [
                  #     (f"constraints/{t}_step_loss", make_n_step_loss(t, full_rollout_loss, batches)(params)) for t in range(1, len(params.theta))
              ]
    # metrics.append(("train/metrics_time", time.time() - metrics_time))
    for tag, value in metrics:
        config.tb.add_scalar(tag, float(value), outer_iter)


def initialize():
    # train_images, train_labels, _, _ = datasets.mnist()
    train_x, train_y, _, _ = datasets.iris()
    dataset_size = train_x.shape[0]
    batch_size = min(config.batch_size, train_x.shape[0])

    def gen_batches() -> (np.ndarray, np.ndarray, List[np.int_]):
        rng = npr.RandomState(0)
        while True:
            indices = np.array(rng.randint(low=0, high=dataset_size, size=(batch_size,)))
            images = np.array(train_x[indices, :])
            labels = np.array(train_y[indices, :])
            yield TaskParameters(images, labels, indices)

    batches = gen_batches()

    blocks_init, model = make_block_net(num_outputs=train_y.shape[1])
    params = init_params(blocks_init, model, train_x)
    return batches, model, params, train_x, train_y


def init_params(blocks_init, model, train_x):
    rng_key = jax.random.PRNGKey(0)
    theta = []
    output_shape = train_x.shape

    for init in blocks_init:
        output_shape, init_params = init(rng_key, output_shape)
        theta.append(init_params)

    y = time_march(train_x, model, theta)
    x = y[:-1]
    return ConstrainedParameters(theta, x)

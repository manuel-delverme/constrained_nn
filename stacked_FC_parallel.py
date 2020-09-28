# train_x, train_y, model, theta, x, y
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
import utils
from network import make_block_net
from utils import ConstrainedParameters, TaskParameters, make_n_step_loss, full_rollout, time_march, train_accuracy


def main():
    batch_gen, model, params, train_x, train_y = initialize()

    def full_rollout_loss(theta: List[np.ndarray], batch):
        train_x, batch_train_y, _indices = batch
        pred_y = full_rollout(train_x, model, theta)
        return np.linalg.norm(pred_y - batch_train_y, 2)

    onestep = make_n_step_loss(1, full_rollout_loss, batch_gen)

    def equality_constraints(params, task):
        theta, x = params
        task_x, _, task_indices = task

        # Layer 1 -> 2
        defects = [model[0](theta[0], task_x) - x[0][task_indices, :], ]

        # Layer 2 onward
        for t in range(len(x) - 1):
            block_x = x[t][task_indices, :]
            block_x = jax.lax.stop_gradient(block_x)

            block_y = x[t + 1][task_indices, :]
            block_y_hat = model[t + 1](theta[t + 1], block_x)
            # defects.append(block_y - jax.lax.stop_gradient(block_y_hat))
            defects.append(block_y_hat - block_y)
            # defects.append(0.)
        return tuple(defects), task_indices

    init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(
        func=onestep,
        # objective_function=lambda params: utils.make_full_rollout_loss(full_rollout_loss, batch_gen)(params),
        equality_constraints=equality_constraints
    )
    initial_values = init_mult(params, (train_x, train_y, np.arange(train_x.shape[0])))
    optimizer_init, optimizer_update, optimizer_get_params = fax.competitive.extragradient.adam_extragradient_optimizer(
        betas=(config.adam1, config.adam2), step_size=config.lr, weight_norm=config.weight_norm)
    opt_state = optimizer_init(initial_values)

    @jax.jit
    def update(i, opt_state):
        grad_fn = jax.grad(lagrangian, (0, 1))
        return optimizer_update(i, grad_fn, opt_state)

    print("optimize()")

    for iter_num in tqdm.trange(config.num_epochs):
        opt_state = update(iter_num, opt_state)

        if iter_num % config.eval_every == 0:
            params = optimizer_get_params(opt_state)
            update_metrics(batch_gen, equality_constraints, full_rollout_loss, model, params, iter_num, train_x, train_y)

    trained_params = optimizer_get_params(opt_state)
    return trained_params


def update_metrics(_batches, equality_constraints, full_rollout_loss, model, params, outer_iter, train_x, train_y):
    params, multipliers = params
    # _train_x, _train_y, _indices = next(batches)
    # metrics_time = time.time()
    fullbatch = train_x, train_y, np.arange(train_x.shape[0])
    h, _task = equality_constraints(params, next(_batches))
    loss = full_rollout_loss(params.theta, next(_batches))

    # def b():
    #     while True:
    #         yield fullbatch

    # batches = b()

    metrics = [
                  ("train_accuracy", train_accuracy(train_x, train_y, model, params.theta)),
                  ("train/sampled_loss", loss),
                  # ("train/full_rollout_loss", full_rollout_loss(params.theta, next(_batches))),
                  # ("train/1step_loss", make_n_step_loss(1, full_rollout_loss, batches)(params)[0]),
                  # ("train/multipliers", np.linalg.norm(multipliers, 1)),
                  # ("train/update_time", time.time() - update_time),
              ] + [
                  (f"constraints/multipliers_l1_{idx}", np.linalg.norm(mi, 1)) for idx, mi in enumerate(multipliers)
              ] + [
                  (f"constraints/defects_l1_{idx}", np.linalg.norm(hi, 1)) for idx, hi in enumerate(h)
              ] + [
                  (f"train/sampled_{t}_step_accuracy", utils.n_step_accuracy(*next(_batches), model, params, t)) for t in range(1, len(params.theta))
                  # ] + [
                  #      (f"constraints/{t}_step_loss", make_n_step_loss(t, full_rollout_loss, batches)(params)) for t in range(1, len(params.theta))
              ]
    # metrics.append(("train/metrics_time", time.time() - metrics_time))
    for tag, value in metrics:
        config.tb.add_scalar(tag, float(value), outer_iter)


def initialize():
    if config.dataset == "mnist":
        train_x, train_y, _, _ = datasets.mnist()
    elif config.dataset == "iris":
        train_x, train_y, _, _ = datasets.iris()
    else:
        raise ValueError

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

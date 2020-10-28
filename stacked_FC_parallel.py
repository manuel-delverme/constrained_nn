from typing import List

import fax
import fax.competitive.extragradient
import fax.constrained
import fax.math
import jax
import jax.experimental.optimizers
import jax.lax
import jax.nn.initializers
import jax.numpy as np
import jax.ops
import jax.tree_util
import numpy.random as npr
import tqdm

import config
import datasets
import utils
from metrics import update_metrics
from network import make_block_net
from utils import ConstrainedParameters, forward_prop, Batch


def make_losses(model):
    def full_rollout_loss(theta: List[np.ndarray], batch: Batch):
        batch_x, batch_y, _indices = batch
        pred_y = forward_prop(batch_x, model, theta)
        return -np.mean(np.sum(pred_y * batch_y, axis=1))

    def one_step_loss(params: ConstrainedParameters, batch: Batch) -> float:
        a_T = config.state_fn(params.x[-1][batch.indices, :])
        pred_y = forward_prop(a_T, model[-1:], params.theta[-1:])
        return -np.mean(np.sum(pred_y * batch.y, axis=1))

    def equality_constraints(params: ConstrainedParameters, batch: Batch) -> (np.array, Batch):
        theta, x = params
        del params
        a_0, _, batch_indices = batch
        a = [a_0, ]
        for xi in x:
            a.append(config.state_fn(xi[batch_indices, :]))

        defects = []
        for t in range(0, len(x)):
            defects.append(
                model[t](theta[t], a[t], ) - a[t + 1]
            )
        return tuple(defects)

    return full_rollout_loss, one_step_loss, equality_constraints


def main():
    batch_gen, model, initial_parameters, full_batch = initialize()
    full_rollout_loss, loss_function, equality_constraints = make_losses(model)

    init_multipliers, lagrangian, get_x = fax.constrained.make_lagrangian(func=loss_function, equality_constraints=equality_constraints)
    initial_values = init_multipliers(initial_parameters, full_batch)

    optimizer_init, optimizer_update, optimizer_get_params = fax.competitive.extragradient.adam_extragradient_optimizer(
        betas=(config.adam1, config.adam2),
        step_sizes=(config.lr_x, config.lr_y),
        weight_norm=config.weight_norm,
        use_adam=config.use_adam,
        grad_clip=config.grad_clip,
    )
    opt_state = optimizer_init(initial_values)

    @jax.jit
    def update(i, opt_state_):
        grad_fn = jax.grad(
            lambda *args: lagrangian(*args, next(batch_gen)),
            (0, 1))
        return optimizer_update(i, grad_fn, opt_state_)

    next_eval = 0
    rng_key = jax.random.PRNGKey(0)

    for iter_num in tqdm.trange(config.num_epochs):
        if next_eval == iter_num:
            params = optimizer_get_params(opt_state)
            update_metrics(lagrangian, make_losses, model, params, iter_num, full_batch)

            rng_key, k_out = jax.random.split(rng_key)
            next_eval += int(config.eval_every + jax.random.randint(k_out, (1,), 0, config.eval_every // 100))

        opt_state = update(iter_num, opt_state)

    trained_params = optimizer_get_params(opt_state)
    return trained_params


def initialize(blocks=False):
    if config.dataset == "mnist":
        train_x, train_y, _, _ = datasets.mnist()
    elif config.dataset == "iris":
        train_x, train_y, _, _ = datasets.iris()
    else:
        raise ValueError

    dataset_size = train_x.shape[0]
    batch_size = min(config.batch_size, train_x.shape[0])

    num_train = train_x.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def gen_batches() -> (np.ndarray, np.ndarray, List[np.int_]):
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(dataset_size)
            for i in range(num_batches):
                indices = perm[i * batch_size:(i + 1) * batch_size]
                images = np.array(train_x[indices, :])
                labels = np.array(train_y[indices, :])
                yield Batch(images, labels, indices)

    batches = gen_batches()
    blocks_init, model = make_block_net(train_y.shape[1], blocks)
    rng_key = jax.random.PRNGKey(0)
    theta = []
    output_shape = train_x.shape

    for init in blocks_init:
        rng_key, k_out = jax.random.split(rng_key)
        output_shape, init_params = init(k_out, output_shape)
        theta.append(init_params)

    x = utils.time_march(train_x, model, theta)
    params = ConstrainedParameters(theta, x[:-1])
    return batches, model, params, Batch(train_x, train_y, np.arange(train_x.shape[0])), num_batches

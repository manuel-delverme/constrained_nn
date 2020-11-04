from typing import List, Tuple, Optional

from jax.config import config as j_config

j_config.update("jax_enable_x64", True)

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
from metrics import update_metrics
from network import make_block_net
from utils import ConstrainedParameters, forward_prop, Batch, LagrangianParameters


def make_losses(model):
    def full_rollout_loss(theta: List[np.ndarray], batch: Batch):
        batch_x, batch_y, _indices = batch
        pred_y = forward_prop(batch_x, model, theta)
        return -np.mean(np.sum(pred_y * batch_y, axis=1))

    def last_layer_loss(params: LagrangianParameters, batch: Batch) -> float:
        x_n = params.constr_params.x[-1]
        a_T = config.state_fn(x_n[batch.indices, :])
        pred_y = forward_prop(a_T, model[-1:], params.constr_params.theta[-1:])
        return -np.mean(np.sum(pred_y * batch.y, axis=1))

    def equality_constraints(params: LagrangianParameters, batch: Batch) -> (np.array, Batch):
        a_0, _, batch_indices = batch
        a = [a_0, ]
        for xi in params.constr_params.x:
            a.append(config.state_fn(xi[batch_indices, :]))

        defects = []
        for t in range(0, len(params.constr_params.x)):
            defects.append(
                model[t](params.constr_params.theta[t], a[t], ) - a[t + 1]
            )
        return tuple(defects)

    return full_rollout_loss, last_layer_loss, equality_constraints


def main():
    batch_gen, model, initial_parameters, full_batch, num_batches = initialize(blocks=[sum(config.blocks), ])
    optimizer_init, optimizer_update, optimizer_get_params = jax.experimental.optimizers.sgd(step_size=config.lr_y, )
    opt_state = optimizer_init(initial_parameters)

    def full_rollout_loss(theta: List[np.ndarray], batch: Batch):
        batch_x, batch_y, _indices = batch
        pred_y = forward_prop(batch_x, model, theta)
        return -np.mean(np.sum(pred_y * batch_y, axis=1))

    @jax.jit
    def update(i, opt_state_, batch):
        grad_fn = jax.grad(full_rollout_loss, 0)
        return optimizer_update(i, grad_fn, opt_state_, batch)

    next_eval = 0
    rng_key = jax.random.PRNGKey(0)

    for iter_num in tqdm.trange(config.num_epochs):
        if next_eval == iter_num:
            params = optimizer_get_params(opt_state)
            update_metrics(make_losses, model, params, iter_num, full_batch)

            rng_key, k_out = jax.random.split(rng_key)
            next_eval += int(config.eval_every + jax.random.randint(k_out, (1,), 0, config.eval_every // 100))

        opt_state = update(iter_num, opt_state, next(batch_gen))

    trained_params = optimizer_get_params(opt_state)
    return trained_params


def init_opt_problem():


def initialize(blocks: Optional[list] = None) -> Tuple[object, object, ConstrainedParameters, object, object]:
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

    return batches, model, theta, Batch(train_x, train_y, np.arange(train_x.shape[0])), num_batches


if __name__ == '__main__':
    main()

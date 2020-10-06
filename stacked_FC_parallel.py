# train_x, train_y, model, theta, x, y
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
import tqdm

import config
import datasets
from metrics import update_metrics
from network import make_block_net
from utils import ConstrainedParameters, forward_prop


def main():
    model, params, train_x, train_y = initialize()

    def full_rollout_loss(theta: List[np.ndarray]):
        pred_y = forward_prop(train_x, model, theta)
        return -np.mean(np.sum(pred_y * train_y, axis=1))

    def loss_function(params):
        theta, activations = params
        x_n = config.state_fn(activations[-1])
        pred_y = forward_prop(x_n, model[-1:], theta[-1:])
        return -np.mean(np.sum(pred_y * train_y, axis=1))

    def equality_constraints(params):
        theta, x = params
        # Layer 1 -> 2
        h0 = model[0](theta[0], train_x) - config.state_fn(x[0])
        defects = [h0, ]

        # Layer 2 onward
        for t in range(len(x) - 1):
            block_x = config.state_fn(x[t])  # grad x
            block_y = config.state_fn(x[t + 1])  # grad x_t+1
            block_y_hat = model[t + 1](theta[t + 1], block_x)  # grad theta

            defects.append(block_y_hat - block_y)
        return tuple(defects)

    init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(
        func=loss_function,
        equality_constraints=equality_constraints
    )

    initial_values = init_mult(params)
    optimizer_init, optimizer_update, optimizer_get_params = fax.competitive.extragradient.adam_extragradient_optimizer(
        betas=(config.adam1, config.adam2), step_sizes=(config.lr_x, config.lr_y), weight_norm=config.weight_norm, use_adam=config.use_adam)
    opt_state = optimizer_init(initial_values)

    @jax.jit
    def update(i, opt_state):
        grad_fn = jax.grad(lagrangian, (0, 1))
        return optimizer_update(i, grad_fn, opt_state)

    print("optimize()")

    for iter_num in tqdm.trange(config.num_epochs):
        if iter_num % config.eval_every == 0:
            params = optimizer_get_params(opt_state)
            update_metrics(lagrangian, equality_constraints, full_rollout_loss, loss_function, model, params, iter_num, train_x, train_y)

        opt_state = update(iter_num, opt_state)

    trained_params = optimizer_get_params(opt_state)
    return trained_params


def initialize():
    if config.dataset == "mnist":
        train_x, train_y, _, _ = datasets.mnist()
    elif config.dataset == "iris":
        train_x, train_y, _, _ = datasets.iris()
    else:
        raise ValueError

    blocks_init, model = make_block_net(num_outputs=train_y.shape[1])
    rng_key = jax.random.PRNGKey(0)
    theta = []
    x = []
    output_shape = train_x.shape

    x_init = jax.nn.initializers.xavier_normal()
    for init in blocks_init:
        rng_key, k_out = jax.random.split(rng_key)
        output_shape, init_params = init(k_out, output_shape)

        theta.append(init_params)

        rng_key, k_out = jax.random.split(rng_key)
        xi = x_init(k_out, output_shape)
        x.append(xi)

    # y = time_march(train_x, model, theta)
    # x = []
    # for xi in y[:-1]:
    #     x.append(np.arctanh(np.clip(xi, -.9, .9)))
    params = ConstrainedParameters(theta, x[:-1])
    return model, params, train_x, train_y

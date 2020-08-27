# train_x, train_y, model, theta, x, y
import collections
from typing import List

import fax
import fax.competitive.extragradient
import jax
import jax.experimental.optimizers
import jax.experimental.stax as stax
import jax.lax
import jax.numpy as np
import jax.ops
import jax.tree_util
import matplotlib.pyplot as plt
import numpy as onp

import config
from main_fax import load_dataset

print("Imported")


class ConstrainedParameters(collections.namedtuple("ConstrainedParameters", "theta x")):
    def __sub__(self, other):
        return jax.tree_util.tree_multimap(lambda _a, _b: _a - _b, self, other)

    def __add__(self, other):
        return jax.tree_util.tree_multimap(lambda _a, _b: _a + _b, self, other)


def block(out_dim, final_activation):
    return stax.serial(
        stax.Dense(32, ),
        stax.LeakyRelu,
        stax.Dense(out_dim, ),
        final_activation,
    )


def make_block_net(num_classes):
    return zip(*[
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(32, stax.LeakyRelu),
        block(num_classes, stax.Softmax),
    ])


def time_march(train_x, model, theta):
    y = []
    x_t = train_x
    for block, theta_t in zip(model, theta):
        x_t = block(theta_t, x_t)
        y.append(x_t)
    return y


def main():
    num_outputs, train_x, train_y, test_x, test_y = load_dataset(normalize=True)
    print("Dataset loaded")

    input_shape = train_x.shape
    batch_size = 32
    lr = 1e-4
    rng_key = jax.random.PRNGKey(0)

    blocks_init, model = make_block_net(num_outputs)
    params = init_params(batch_size, blocks_init, model, input_shape, rng_key, train_x)

    iters = 100000

    def full_rollout_loss(theta: List[np.ndarray], batch):
        train_x, train_y, _indices = batch
        # train_x, train_y, indices = next(batches)

        y = time_march(train_x, model, theta)
        predicted = y[-1]
        error = train_y - predicted
        return np.linalg.norm(error, 2)

    def make_n_step_loss(n):
        def n_step_loss(params):
            theta, activations = params
            train_x, train_y, indices = next(batches)
            batch = activations[-n][indices], train_y, indices
            return full_rollout_loss(theta[-n:], batch)

        return n_step_loss

    def equality_constraints(params):
        theta, activations = params
        train_x, train_y, indices = next(batches)
        batch_activations = []
        for layer_act in activations:
            # xi = jax.take(layer_x, indices.reshape(-1, 1), (0, ))
            xi = layer_act[indices]
            batch_activations.append(xi)
        # x = x[indices]

        defects = []
        x = batch_activations[:-1]
        y = batch_activations[1:]

        # this could be replaced by vmap(np.linalg.norm(split_variables_batch - vmap(block(split_left)))
        # vmap(function, input_batch_arguments, output_batch_arguments)
        # time_march(train_x, model, theta)
        for x_t, block_t, theta_t, y_t in zip(x, model, theta, y):
            # jax.lax.stop_gradient() ?
            defects.append(y_t - block_t(theta_t, x_t))
        return np.hstack(defects), indices

    def gen_batches() -> (np.ndarray, np.ndarray, List[np.int_]):
        rng = onp.random.RandomState(0)
        while True:
            indices = rng.randint(train_x.shape[0], size=(batch_size,))
            images = train_x[indices]
            labels = train_y[indices]
            yield images, labels, indices

    batches = gen_batches()

    print("Constrained setting", config.constrained)
    if config.constrained:
        onestep = make_n_step_loss(1)
        init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(
            lambda params: -onestep(params),
            # lambda params: -full_rollout_loss(params.theta, next(batches)),
            equality_constraints,
            # lambda params: 0.,
        )
        initial_values = init_mult(params)

        optimizer_init, optimizer_update, optimizer_get_params = fax.competitive.extragradient.adam_extragradient_optimizer(step_size=lr)
        opt_state = optimizer_init(initial_values)

        @jax.jit
        def update(i, opt_state):
            grad_fn = jax.grad(lagrangian, (0, 1))
            return optimizer_update(i, grad_fn, opt_state)

        print("optimize()")
        for outer_iter in range(iters):
            opt_state = update(outer_iter, opt_state)
            if outer_iter % 10 == 0:
                params = optimizer_get_params(opt_state)
                params, multipliers = params
                _train_x, _train_y, _indices = next(batches)
                metrics = [
                              # ("train/x_err", x_err),
                              # ("train/theta_err", theta_err),
                              ("train/train_accuracy", train_accuracy(_train_x, _train_y, model, params.theta)),
                              ("train/train_loss", full_rollout_loss(params.theta, next(batches))),
                              ("train/1step_loss", make_n_step_loss(1)(params)),
                              ("train/multipliers", np.linalg.norm(multipliers, 2)),
                              # ("train/2step_accuracy", n_step_loss_fn(2)),
                              # ("train/3step_accuracy", NotImplemented)
                          ] + [
                              (f"constraints/defects_{idx}", np.linalg.norm(equality_constraints(params)[0][idx], 2)) for idx in range(len(params.theta))
                          ] + [
                              (f"rollouts/{idx}_step_prediction", make_n_step_loss(idx)(params)) for idx in range(len(params.theta))
                          ]
                push_metrics(outer_iter, metrics)
            if outer_iter % 1000 == 0:
                params = optimizer_get_params(opt_state)
                params, multipliers = params
                y = time_march(train_x, model, params.theta)
                x = [train_x, *y[:-1]]
                params = ConstrainedParameters(params.theta, x)
                initial_values = init_mult(params)
                opt_state = optimizer_init(initial_values)

    else:
        opt_init, opt_update, get_params = jax.experimental.optimizers.momentum(lr, mass=0.9)
        train_loss_function = lambda theta: full_rollout_loss(theta, next(batches))

        @jax.jit
        def update(i, opt_state):
            theta = get_params(opt_state)
            return opt_update(i, jax.grad(train_loss_function, 0)(theta), opt_state)

        opt_state = opt_init(params.theta)
        for outer_iter in range(iters):
            opt_state = update(outer_iter, opt_state)
            params = get_params(opt_state)

            _train_x, _train_y, _indices = next(batches)
            metrics = [
                # ("train/x_err", x_err),
                # ("train/theta_err", theta_err),
                ("train/train_accuracy", train_accuracy(_train_x, _train_y, model, params)),
                ("train/train_loss", train_loss_function(params)),
                # ("train/1step_loss", make_n_step_loss(1)(params)),
                # ("train/2step_accuracy", n_step_loss_fn(2)),
                # ("train/3step_accuracy", NotImplemented)
            ]
            push_metrics(outer_iter, metrics)

        trained_params = get_params(opt_state)
    return trained_params


def init_params(batch_size, blocks_init, model, input_shape, rng_key, train_x):
    theta = []
    for init in blocks_init:
        _, init_params = init(rng_key, input_shape)
        theta.append(init_params)
        input_shape = (batch_size, *init_params[-2][-1].shape)  # use bias shape since it's dense layers
    y = time_march(train_x, model, theta)
    x = [train_x, *y[:-1]]
    # x = []
    # for xi in [train_x, *y[:-1]]:
    #     rng_key, layer_rng = jax.random.split(rng_key)
    #     x.append(jax.random.uniform(layer_rng, xi.shape))
    return ConstrainedParameters(theta, x)


# train_x: np.ndarray, train_y: np.ndarray, model: List[Callable],
def train_accuracy(train_x, train_y, model, theta):
    target_class = np.argmax(train_y, axis=-1)
    x_ = time_march(train_x, model, theta)
    logits = x_[-1]
    predicted_class = np.argmax(logits, axis=-1)
    return np.mean(predicted_class == target_class)


def push_metrics(outer_iter, metrics):
    for tag, value in metrics:
        config.tb.add_scalar(tag, float(value), outer_iter)
        print(tag, value)


if __name__ == "__main__":
    try:
        with jax.disable_jit():
            main()
    except KeyboardInterrupt:
        plt.close("all")

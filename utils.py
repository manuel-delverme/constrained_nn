import collections

import fax
import fax.competitive.extragradient
import jax
import jax.experimental.optimizers
import jax.lax
import jax.tree_util
from jax import tree_util, numpy as np
from matplotlib import pyplot as plt


def time_march(train_x, model, theta):
    y = []
    x_t = train_x
    for block, theta_t in zip(model, theta):
        x_t = block(theta_t, x_t)
        y.append(x_t)
    return y


def full_rollout(train_x, model, theta):
    x_t = train_x
    for block, theta_t in zip(model, theta):
        x_t = block(theta_t, x_t)
    return x_t


def division_constant(constant):
    def divide(a):
        return tree_util.tree_multimap(lambda _a: _a / constant, a)

    return divide


def multiply_constant(constant):
    def multiply(a):
        return tree_util.tree_multimap(lambda _a: _a * constant, a)

    return multiply


division = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a / _b, _a, _b)
add = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a + _b, _a, _b)
sub = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a - _b, _a, _b)


def mul(_a, _b):
    return tree_util.tree_multimap(jax.lax.mul, _a, _b)


# mul = lambda a, b: tree_util.tree_multimap(lax.mul(a, b), a, b)
square = lambda _a: tree_util.tree_map(np.square, _a)


def sgd_solve(lagrangian, convergence_test, get_x, initial_values, max_iter=100000000, metrics=(), lr=None):
    optimizer_init, optimizer_update, optimizer_get_params = adam_optimizer(lr)

    @jax.jit
    def update(i, opt_state):
        grad_fn = jax.grad(lagrangian, (0, 1))
        return optimizer_update(i, grad_fn, opt_state)

    solution, history = fax.loop.fixed_point_iteration(
        init_x=optimizer_init(initial_values),
        func=update,
        convergence_test=convergence_test,
        max_iter=max_iter,
        get_params=optimizer_get_params,
        metrics=metrics,
    )
    return *get_x(solution), history


# def adam_optimizer(step_size, betas=(config.adam1, config.adam2), eps=config.adam_eps) -> (Callable, Callable, Callable):
#     step_size = jax.experimental.optimizers.make_schedule(step_size)
#
#     def init(init_values):
#         # Exponential moving average of gradient values
#         # exp_avg = np.zeros_like(init_values)
#         exp_avg = jax.tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), init_values)
#
#         # Exponential moving average of gradient values
#         exp_avg_sq = jax.tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), init_values)
#
#         return init_values, (exp_avg, exp_avg_sq)
#
#     def update(step, grad_fns, state):
#         (x0, y0), grad_state = state
#         step_sizes = step_size(step)
#
#         (delta_x, delta_y), grad_state = fax.competitive.extragradient.adam_step(betas, eps, step_sizes, grad_fns, grad_state, x0, y0, step)
#         x1 = sub(x0, delta_x)
#         y1 = add(y0, delta_y)
#         return (x1, y1), grad_state
#
#     def get_params(state):
#         x, _ = state
#         return x
#
#     return init, update, get_params


def plot_model(model, trainX, trainY, title, i):
    # xs = [0, 1, 2]

    # block_outs = [*model.split_variables, trainY]
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.1, 3.1)
    xs = np.array((
        trainX[0][0],
        model.split_variables[0][0][0],
        trainY[0][0],
    ))
    ax.scatter(range(len(xs)), xs)

    for t, (block, x_t) in enumerate(zip(model.blocks, xs)):
        x_t1 = block(x_t)
        ax.plot([t, t + 1], [x_t, x_t1])
    fig.savefig(f"plots/{i}.png")
    fig.show()


class ConstrainedParameters(collections.namedtuple("ConstrainedParameters", "theta x")):
    def __sub__(self, other):
        return jax.tree_util.tree_multimap(lambda _a, _b: _a - _b, self, other)

    def __add__(self, other):
        return jax.tree_util.tree_multimap(lambda _a, _b: _a + _b, self, other)


class TaskParameters(collections.namedtuple("TaskParameters", "x y idx")):
    pass


class TaskParameters(tuple):
    def __new__(cls, *args, **kwargs):
        return tuple(args)


class LagrangianParameters(collections.namedtuple("LagrangianParameters", "constr_params multipliers")):
    pass


# class LagrangianParameters(tuple):
#     def __new__(cls, *args, **kwargs):
#         return tuple(args)


def train_accuracy(train_x, train_y, model, theta):
    logits = full_rollout(train_x, model, theta)
    predicted_class = np.argmax(logits, axis=-1)
    target_class = np.argmax(train_y, axis=-1)
    return np.mean(predicted_class == target_class)


def n_step_accuracy(_train_x, train_y, indices, model, params, n):
    assert 0 < n < len(model)
    return train_accuracy(
        params.x[-n][indices, :],
        train_y,
        model[-n:],
        params.theta[-n:],
    )


def make_n_step_loss(n, full_rollout_loss, batches):
    assert n > 0

    def n_step_loss(params):
        theta, activations = params
        x0 = next(batches)
        _, train_y, indices = x0
        x_n = jax.lax.stop_gradient(activations[-n][indices])
        theta_n_T = jax.lax.stop_gradient(theta[-n:])

        x_n = (x_n, train_y, indices)
        return full_rollout_loss(theta_n_T, x_n), x0

    return n_step_loss


def make_full_rollout_loss(full_rollout_loss, batches):
    def n_step_loss(params):
        batch = next(batches)
        return full_rollout_loss(params.theta, batch), batch

    return n_step_loss

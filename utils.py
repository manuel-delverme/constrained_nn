import collections

import jax
import jax.experimental.optimizers
import jax.lax
import jax.tree_util
from jax import tree_util, numpy as np

import config


def time_march(x0, model, theta):
    y = []
    x_t = x0
    for block, theta_t in zip(model, theta):
        x_t = block(theta_t, x_t)
        y.append(x_t)
    return y


def one_step(x0, x, model, theta):
    y = []
    for x_t, block, theta_t in zip([x0, *x], model, theta):
        y_t = block(theta_t, config.state_fn(x_t))
        y.append(y_t)
    return y


def forward_prop(x0, model, theta):
    x_t = config.state_fn(x0)
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


class Batch(collections.namedtuple("Batch", "x y indices")):
    pass


class ConstrainedParameters(collections.namedtuple("ConstrainedParameters", "theta x")):
    def __sub__(self, other):
        return jax.tree_util.tree_multimap(lambda _a, _b: _a - _b, self, other)

    def __add__(self, other):
        return jax.tree_util.tree_multimap(lambda _a, _b: _a + _b, self, other)


class LagrangianParameters(collections.namedtuple("LagrangianParameters", "constr_params multipliers")):
    pass


# class LagrangianParameters(tuple):
#     def __new__(cls, *args, **kwargs):
#         return tuple(args)


def train_acc(train_x, train_y, model, theta):
    logits = forward_prop(train_x, model, theta)
    predicted_class = np.argmax(logits, axis=-1)
    target_class = np.argmax(train_y, axis=-1)
    return -np.mean(np.sum(logits * train_y, axis=1)), np.mean(predicted_class == target_class)


def n_step_acc(train_x, train_y, model, params, n):
    assert 0 < n < len(model) + 1
    return train_acc(
        [train_x, *[config.state_fn(xi) for xi in params.x]][-n],
        # config.state_fn(params.x[-n]),
        train_y,
        model[-n:],
        params.theta[-n:],
    )
#
#
# def train_loss(train_x, train_y, model, theta):
#     logits = forward_prop(train_x, model, theta)
#     # predicted_class = np.argmax(logits, axis=-1)
#     # target_class = np.argmax(train_y, axis=-1)
#     # return np.mean(predicted_class == target_class)
#     return -np.mean(np.sum(logits * train_y, axis=1))
#

# def n_step_loss(train_x, train_y, model, params, n):
#     assert 0 < n < len(model) + 1
#     return train_loss(
#         [train_x, *[config.state_fn(xi) for xi in params.x]][-n],
#         # config.state_fn(params.x[-n]),
#         train_y,
#         model[-n:],
#         params.theta[-n:],
#     )

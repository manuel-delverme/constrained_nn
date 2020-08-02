from typing import Callable

import fax
import fax.competitive.extragradient
import jax
import jax.experimental.optimizers
import jax.lax
import jax.numpy as np
from jax import tree_util

import config


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


def adam_optimizer(step_size, betas=config.adam_betas, eps=config.adam_eps) -> (Callable, Callable, Callable):
    step_size = jax.experimental.optimizers.make_schedule(step_size)

    def init(init_values):
        # Exponential moving average of gradient values
        # exp_avg = np.zeros_like(init_values)
        exp_avg = jax.tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), init_values)

        # Exponential moving average of gradient values
        exp_avg_sq = jax.tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), init_values)

        return init_values, (exp_avg, exp_avg_sq)

    def update(step, grad_fns, state):
        (x0, y0), grad_state = state
        step_sizes = step_size(step)

        (delta_x, delta_y), grad_state = fax.competitive.extragradient.adam_step(betas, eps, step_sizes, grad_fns, grad_state, x0, y0, step)
        x1 = sub(x0, delta_x)
        y1 = add(y0, delta_y)
        return (x1, y1), grad_state

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params

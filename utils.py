from typing import Callable

import fax
import fax.competitive.extragradient
import jax
import jax.experimental.optimizers
import jax.lax
import jax.numpy as np
from jax import tree_util


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

    fixpoint_fn = fax.loop._debug_fixed_point_iteration if fax.config.DEBUG else fax.loop.fixed_point_iteration
    solution = fixpoint_fn(
        init_x=optimizer_init(initial_values),
        func=update,
        convergence_test=convergence_test,
        max_iter=max_iter,
        get_params=optimizer_get_params,
        metrics=metrics,
    )
    return get_x(solution)


def adam_optimizer(step_size, betas=(0.3, 0.99), eps=1e-8) -> (Callable, Callable, Callable):
    """Provides an optimizer interface to the extra-gradient method

    We are trying to find a pair (x*, y*) such that:

    f(x*, y) ≤ f(x*, y*) ≤ f(x, y*), ∀ x ∈ X, y ∈ Y

    where X and Y are closed convex sets.

    Args:
        init_values:
        step_size_x (float): x learning rate,
        step_size_y: (float): y learning rate,
        f: Saddle-point function
        convergence_test:  TODO
        max_iter:  TODO
        batched_iter_size:  TODO
        unroll:  TODO

        betas (Tuple[float, float]): coefficients used for computing running averages of gradient and its square.
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        ams_grad (boolean, optional): whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_

    """

    step_size = jax.experimental.optimizers.make_schedule(step_size)

    def init(init_values):
        # Exponential moving average of squared gradient values

        # model, multipliers = init_values

        # h = jax.eval_shape(equality_constraints, params, *args, **kwargs)
        # multipliers = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), model)

        # assert len(x0.shape) == (len(y0.shape) == 1 or not y0.shape)
        # model_params = np.hstack([mi.flatten() for mi in model])
        # multipliers = multipliers.flatten()

        # init_values = np.concatenate((model_params, multipliers))

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

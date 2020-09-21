# train_x, train_y, model, theta, x, y
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
import tqdm
from jax.experimental import stax as stax

import config
from utils import full_rollout, ConstrainedParameters


def make_block_net(num_outputs):
    return zip(*[
        stax.serial(
            stax.Dense(num_outputs),
        ),
    ])


def main():
    blocks_init, model = make_block_net(num_outputs=2)

    rng_key = jax.random.PRNGKey(0)
    theta = []
    train_x = np.ones((1, 1))
    output_shape = train_x.shape

    for init in blocks_init:
        output_shape, init_params = init(rng_key, output_shape)
        theta.append(init_params)
    params = ConstrainedParameters(theta, None)

    player_1_obj = lambda params: full_rollout(train_x, model, params.theta)[0, 0]

    def h(params):
        indices = (0,)
        constr = [
            full_rollout(train_x, model, params.theta)[0, 1].reshape(1, 1),
        ]
        return np.array(constr), indices

    init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(player_1_obj, h)
    initial_values = init_mult(params)

    optimizer_init, optimizer_update, optimizer_get_params = fax.competitive.extragradient.adam_extragradient_optimizer(step_size=config.lr)
    opt_state = optimizer_init(initial_values)

    @jax.jit
    def update(i, opt_state):
        grad_fn = jax.grad(lagrangian, (0, 1))
        return optimizer_update(i, grad_fn, opt_state)

    for iter_num in tqdm.trange(config.num_epochs):
        opt_state = update(iter_num, opt_state)

        lagrangian_params = optimizer_get_params(opt_state)
        params = lagrangian_params[0]
        update_metrics(player_1_obj, h, params, iter_num)

    trained_params = optimizer_get_params(opt_state)
    return trained_params


def update_metrics(p1, h, param, it):
    metrics = [
        ("train/p1", p1(param)),
        ("train/h", h(param)[0]),
    ]
    for tag, value in metrics:
        config.tb.add_scalar(tag, float(value), it)


# with jax.disable_jit():
main()

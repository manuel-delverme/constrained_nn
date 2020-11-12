from typing import List, Tuple, Callable, Any

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
from fax.utils import ConstrainedParameters, Batch, LagrangianParameters

import config
import datasets
import utils
from metrics import update_metrics
from network import make_block_net


def make_losses(model):
    def full_rollout_loss(theta: List[np.ndarray], batch: Batch):
        batch_x, batch_y, _indices = batch
        pred_y = utils.forward_prop(batch_x, model, theta)
        return -np.mean(np.sum(pred_y * batch_y, axis=1))

    def last_layer_loss(params: LagrangianParameters, batch: Batch) -> float:
        x_n = params.constr_params.x[-1]
        a_T = x_n[batch.indices, :]
        pred_y = utils.forward_prop(a_T, model[-1:], params.constr_params.theta[-1:])
        return -np.mean(np.sum(pred_y * batch.y, axis=1))

    def equality_constraints(params: LagrangianParameters, batch: Batch) -> (np.array, Batch):
        a_0, _, batch_indices = batch
        a = [a_0, ]
        for xi in params.constr_params.x:
            a.append(xi[batch_indices, :])

        defects = []
        for t in range(0, len(params.constr_params.x)):
            defects.append(
                model[t](params.constr_params.theta[t], a[t], ) - a[t + 1]
            )
        return tuple(defects)

    return full_rollout_loss, last_layer_loss, equality_constraints


def main():
    full_batch, model, opt_state, optimizer_get_params, lagrangian, optimizer_update, batch_gen, num_batches, test_batch = init_opt_problem()

    @jax.jit
    def update(i, opt_state_, batch):
        grad_fn = jax.grad(lagrangian, 0)
        return optimizer_update(i, grad_fn, opt_state_, batch)

    next_eval = 0
    rng_key = jax.random.PRNGKey(0)

    for iter_num in tqdm.trange(config.num_epochs):
        if next_eval == iter_num:
            params = optimizer_get_params(opt_state)
            update_metrics(lagrangian, make_losses, model, params, iter_num, full_batch, test_batch)

            rng_key, k_out = jax.random.split(rng_key)
            next_eval += int(config.eval_every + jax.random.randint(k_out, (1,), 0, config.eval_every // 100))

        opt_state = update(iter_num, opt_state, next(batch_gen))

    trained_params = optimizer_get_params(opt_state)
    return trained_params


def init_opt_problem() -> Tuple[Batch, List[Callable], Any, Any, Any, Any, object, object, Batch]:
    batch_gen, model, initial_parameters, full_batch, num_batches, test_batch = initialize(config.blocks)
    if not isinstance(initial_parameters, ConstrainedParameters):
        raise TypeError("nah")

    _, last_layer_loss, equality_constraints = make_losses(model)
    init_multipliers, lagrangian, get_x = fax.constrained.make_lagrangian(
        func=last_layer_loss, equality_constraints=equality_constraints)
    initial_parameters = init_multipliers(initial_parameters, full_batch)
    optimizer_init, optimizer_update, optimizer_get_params = fax.competitive.extragradient.adam_extragradient_optimizer(
        betas=(config.adam1, config.adam2),
        step_sizes=(config.lr_theta, config.lr_x, config.lr_y),
        weight_norm=config.weight_norm,
        use_adam=config.use_adam,
        grad_clip=config.grad_clip,
    )
    opt_state = optimizer_init(initial_parameters)
    return full_batch, model, opt_state, optimizer_get_params, lagrangian, optimizer_update, batch_gen, num_batches, test_batch


def initialize(blocks) -> Tuple[object, List[Callable], ConstrainedParameters, Batch, object, Batch]:
    if config.dataset == "mnist":
        train_x, train_y, test_x, test_y = datasets.mnist()
    elif config.dataset == "iris":
        train_x, train_y, test_x, test_y = datasets.iris()
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

    for t, init in enumerate(blocks_init):
        rng_key, k_out = jax.random.split(rng_key)
        print("init block", t)
        output_shape, init_params = init(k_out, output_shape)
        theta.append(init_params)

    x = utils.time_march(train_x, model, theta)
    params = ConstrainedParameters(theta, x[:-1])
    print("init x")
    return batches, model, params, Batch(train_x, train_y, np.arange(train_x.shape[0])), num_batches, Batch(test_x, test_y, np.arange(test_x.shape[0]))

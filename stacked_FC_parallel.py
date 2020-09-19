# train_x, train_y, model, theta, x, y
import itertools
import time
from typing import List

import fax
import fax.competitive.extragradient
import fax.constrained
import fax.math
import jax
import jax.experimental.optimizers
import jax.experimental.stax as stax
import jax.lax
import jax.numpy as np
import jax.ops
import jax.tree_util
import numpy.random as npr
import tqdm

import config
import datasets
from utils import ConstrainedParameters, LagrangianParameters, TaskParameters, sub


def block(out_dim, final_activation):
    return stax.serial(
        stax.Dense(out_dim, ),
        final_activation,
    )


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


# def make_n_step_loss(n, full_rollout_loss):
#     def n_step_loss(batch, params):
#         theta, activations = params
#         train_x, train_y, indices = batch
#         batch = activations[-n][indices], train_y, indices
#         return full_rollout_loss(theta[-n:], batch)
#     return n_step_loss


def main():
    batch_generator, num_batches, equality_constraints, full_rollout_loss, initial_values, onestep, full_batch_loss, full_batch_defect = initialize()

    def lagrangian(task_params, task_multipliers, task):
        defects = equality_constraints(task, task_params)
        # task_multipliers = [mi[task[2], :] for mi in multipliers]
        # return full_rollout_loss(task_params.theta, task) + fax.math.pytree_dot(task_multipliers, defects)
        # return onestep(task, task_params) + fax.math.pytree_dot(task_multipliers, defects)  # + 0.1 * fax.math.pytree_dot(defects, defects)
        return onestep(task, task_params) + fax.math.pytree_dot(task_multipliers, defects)  # + 0.01 * fax.math.pytree_dot(defects, defects)
        # return onestep(task, task_params)#  + 0.1 * fax.math.pytree_dot(defects, defects)
        # return full_rollout_loss(task_params.theta, task) + fax.math.pytree_dot(task_multipliers, defects)

    optimizer_init, optimizer_update, optimizer_get_params = fax.competitive.extragradient.adam_extragradient_optimizer(step_size=config.lr, betas=(config.adam1, config.adam2))
    opt_state = optimizer_init(initial_values)

    @jax.jit
    def update(i, global_opt_state, task):
        def task_lagrangian(params, multipliers):
            return lagrangian(params, multipliers, task)

        grad_fn = jax.grad(task_lagrangian, (0, 1))
        task_x, task_y, task_idx = task
        global_parameters, global_grad_state = global_opt_state
        dataset_size = global_parameters[1][0].shape[0]

        def task_slice_param(global_parameters):
            constr_param, global_multipliers = global_parameters

            def slice_(p):
                assert p.shape[0] == dataset_size
                return p[task_idx, :]

            task_x = jax.tree_util.tree_map(slice_, constr_param.x)
            task_multipliers = [slice_(gi) for gi in global_multipliers]
            return ConstrainedParameters(constr_param.theta, task_x), task_multipliers

        task_opt_state = task_slice_param(global_parameters), tuple([task_slice_param(p) for p in global_grad_state])
        new_local_state = optimizer_update(i, grad_fn, task_opt_state)

        def task_update(globals, sampled):
            global_params, global_multipliers = globals
            local_params, local_multipliers = sampled

            for time_step in range(len(global_params.x)):
                global_params.x[time_step] = jax.ops.index_update(global_params.x[time_step], task_idx, local_params.x[time_step], unique_indices=True)
                global_multipliers[time_step] = jax.ops.index_update(global_multipliers[time_step], jax.ops.index[task_idx, :], local_multipliers[time_step], unique_indices=True)

            return ConstrainedParameters(local_params.theta, global_params.x), global_multipliers

        new_local_parameters, new_local_grad_state = new_local_state

        new_global_parameters = task_update(global_parameters, new_local_parameters)
        new_global_grad_state = tuple([task_update(a, b) for a, b in zip(global_grad_state, new_local_grad_state)])
        return new_global_parameters, new_global_grad_state

    # print("optimize()")
    itercount = itertools.count()
    update_time = time.time()

    for epoch_num in tqdm.trange(config.num_epochs):
        # iters = 100
        iters = num_batches
        for _ in tqdm.trange(iters, disable=True):
            opt_state = update(next(itercount), opt_state, next(batch_generator))

        epoch_time = time.time() - update_time
        # print("Epoch", epoch_num)

        # gc.collect()
        params = optimizer_get_params(opt_state)
        update_metrics(full_batch_defect, full_batch_loss, params, epoch_num, epoch_time)
        update_time = time.time()

        # if outer_iter % 1000 == 0:
        #     global_params = optimizer_get_params(opt_state)
        #     global_params, multipliers = global_params
        #     y = time_march(train_x, predict_fn, global_params.theta)
        #     x = [train_x, *y[:-1]]
        #     global_params = ConstrainedParameters(global_params.theta, x)
        #     initial_values = init_mult(global_params)
        #     opt_state = optimizer_init(initial_values)

    trained_params = optimizer_get_params(opt_state)
    return trained_params


def update_metrics(defects, train_loss, params, outer_iter, epoch_time):
    params, multipliers = params
    metrics_time = time.time()
    l1_loss, accuracy = train_loss(params.theta)

    defects_ = defects(params=params)
    # [defects(params=params)[0][idx], 2)) for idx in range(len(params.theta)]

    metrics = [
                  ("train/train_accuracy", accuracy),
                  ("train/train_l1_loss", l1_loss),
                  # ("train/1step_loss", make_n_step_loss(1, train_loss)(params)),
                  # ("train/multipliers", np.linalg.norm(multipliers, 2)),
                  # ("train/update_time", epoch_time),
              ] + [
                  (f"constraints/defects_{idx}", np.linalg.norm(defects(params=params)[0][idx], 2)) for idx in range(len(params.theta))
              ] + [
                  (f"constraints/multipliers{idx}", np.linalg.norm(multipliers[idx], 2)) for idx in range(len(params.x))
                  # ] + [
                  #     (f"rollouts/{idx}_step_prediction", make_n_step_loss(idx, train_loss, batches)(params)) for idx in range(len(params.theta))
              ]
    # metrics.append(("train/metrics_time", time.time() - metrics_time))
    for tag, value in metrics:
        config.tb.add_scalar(tag, float(value), outer_iter)
        # print(tag, value)


def gen_batches():
    # train_images, train_labels, _, _ = datasets.mnist()
    train_images, train_labels, _, _ = datasets.iris()
    num_train = train_images.shape[0]
    bs = min(config.batch_size, num_train)

    num_complete_batches, leftover = divmod(num_train, bs)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * bs:(i + 1) * bs]
                batch_idx = np.sort(batch_idx)
                yield TaskParameters(train_images[batch_idx, :], train_labels[batch_idx, :], batch_idx)

    batches = data_stream()
    return batches, train_images.shape, train_images, num_batches, train_labels


def initialize():
    batch_generator, input_shape, train_x, num_batches, train_y = gen_batches()
    blocks_init, blocks_predict = zip(*[
        # stax.serial(
        #     # stax.Dense(1024, ),
        #     stax.Dense(64, ),
        #     stax.LeakyRelu,
        # ),
        stax.serial(
            # stax.Dense(1024, ),
            stax.Dense(64, ),
            stax.LeakyRelu,
        ),
        stax.serial(
            stax.Dense(3),
            stax.LogSoftmax
        ),
    ])

    rng_key = jax.random.PRNGKey(0)
    theta = []
    output_shape = (-1, *input_shape[1:])
    for init in blocks_init:
        output_shape, init_params = init(rng_key, output_shape)
        theta.append(init_params)

    y = time_march(train_x, blocks_predict, theta)
    x = y[:-1]
    initial_solution = ConstrainedParameters(theta, x)

    def full_rollout_loss(theta: List[np.ndarray], batch):
        batch_train_x, batch_train_y, _indices = batch

        pred_y = full_rollout(batch_train_x, blocks_predict, theta)
        # error = batch_train_y - pred_y
        # return np.linalg.norm(error, 2)
        return -np.mean(np.sum(batch_train_y * pred_y, axis=1))

    def one_step(batch, params):
        theta, activations = params
        train_x, train_y, indices = batch
        if activations:
            x0 = activations[-1][indices]
        else:
            x0 = train_x
        batch = x0, train_y, indices
        return full_rollout_loss(theta[-1:], batch)

    def equality_constraints(task, params):
        theta, task_activations = params
        train_x, train_y, indices = task
        batch_size = train_x.shape[0]
        assert task_activations[0].shape[0] == batch_size
        # task_activations = []
        # for layer_act in activations:
        #     xi = layer_act[indices]
        #     batch_activations.append(xi)

        defects = []
        layer_inputs = [train_x, *task_activations[:-1]]
        layer_targets = task_activations

        # y = time_march(train_x, blocks_predict, theta)
        for x_t, block_t, theta_t, y_t in zip(layer_inputs, blocks_predict, theta, layer_targets):
           defects.append(np.power(y_t - block_t(theta_t, x_t), 2))
        # return sub(y[:-1], task_activations)
        return defects

    def full_batch_loss(theta: List[np.ndarray]):
        predicted = full_rollout(train_x, blocks_predict, theta)

        target_class = np.argmax(train_y, axis=-1)
        predicted_class = np.argmax(predicted, axis=-1)
        return -np.mean(np.sum(train_y * predicted, axis=1)), np.mean(predicted_class == target_class)

    # full_batch_loss = jax.partial(full_rollout_loss, batch=(train_x, train_y, None))
    full_batch_defect = jax.partial(equality_constraints, task=(train_x, train_y, slice(None, None)))

    initial_values = initialize_lagrangian(equality_constraints, initial_solution, train_x, y)
    return batch_generator, num_batches, equality_constraints, full_rollout_loss, initial_values, one_step, full_batch_loss, full_batch_defect


def initialize_lagrangian(equality_constraints, params, train_x, y):
    task = TaskParameters(train_x, y[-1], np.arange(train_x.shape[0]))
    h = jax.eval_shape(equality_constraints, task, params)
    multipliers = [np.zeros(hi.shape) for hi in h]

    initial_values = LagrangianParameters(params, multipliers)
    return initial_values

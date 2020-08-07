# train_x, train_y, model, theta, x, y
from typing import List

import jax
import jax.experimental.optimizers
import jax.experimental.stax as stax
import jax.lax
import jax.numpy as np
import jax.ops
import jax.tree_util
import matplotlib.pyplot as plt

from main_fax import load_dataset


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

    input_shape = train_x.shape
    batch_size = train_x.shape[0]
    rng_key = jax.random.PRNGKey(0)

    blocks_init, model = make_block_net(num_outputs)

    theta = []
    for init in blocks_init:
        _, init_params = init(rng_key, input_shape)
        theta.append(init_params)
        input_shape = (batch_size, *init_params[-2][-1].shape)  # use bias shape since it's dense layers

    y = time_march(train_x, model, theta)
    x = [train_x, *y[:-1]]
    lr = 1e-4

    opt_init, opt_update, get_params = jax.experimental.optimizers.momentum(lr, mass=0.9)

    def full_rollout_loss(train_x, theta: List[np.ndarray]):
        y = time_march(train_x, model, theta)
        predicted = y[-1]
        error = train_y - predicted
        return np.linalg.norm(error, 2)

    def n_step_loss(theta: List[np.ndarray], x: List[np.ndarray], n):
        x0 = x[-n]
        return full_rollout_loss(x0, theta[-n:])

    @jax.jit
    def update(i, opt_state, train_x):
        theta = get_params(opt_state)
        return opt_update(i, jax.grad(full_rollout_loss, 1)(train_x, theta), opt_state)

    opt_state = opt_init(theta)
    for outer_iter in range(1000):
        opt_state = update(outer_iter, opt_state, train_x)
        theta = get_params(opt_state)
        push_metrics(train_x, train_y, model, theta, x, outer_iter, full_rollout_loss)

    trained_params = get_params(opt_state)
    return trained_params


# train_x: np.ndarray, train_y: np.ndarray, model: List[Callable],
def train_accuracy(train_x, train_y, model, theta):
    target_class = np.argmax(train_y, axis=-1)
    x_ = time_march(train_x, model, theta)
    logits = x_[-1]
    predicted_class = np.argmax(logits, axis=-1)
    return np.mean(predicted_class == target_class)


def push_metrics(train_x, train_y, model, theta, x, outer_iter, loss_fn):
    accuracy = train_accuracy(train_x, train_y, model, theta)
    train_loss = loss_fn(train_x, theta)
    metrics = [
        # ("train/x_err", x_err),
        # ("train/theta_err", theta_err),
        ("train/train_accuracy", accuracy),
        ("train/train_loss", train_loss),
        # ("train/1step_accuracy", NotImplemented),
        # ("train/2step_accuracy", NotImplemented),
        # ("train/3step_accuracy", NotImplemented)
    ]
    for tag, value in metrics:
        # tb.add_scalar(tag, float(value), opt_step)
        print(tag, value)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")

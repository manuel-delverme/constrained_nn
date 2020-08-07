import fax
import jax
import jax.experimental.optimizers
import jax.experimental.stax as stax
import jax.lax
import jax.numpy as np
import jax.ops
import jax.tree_util
import matplotlib.pyplot as plt

from main_fax import load_dataset


class Parameters(jax.collections.namedtuple("Parameters", "theta x")):
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

    def full_rollout_loss(train_x, model, theta):
        y = time_march(train_x, model, theta)
        predicted = y[-1]
        error = train_y - predicted
        return np.linalg.norm(error, 2)

    def n_step_loss(n):
        def loss(params):
            return full_rollout_loss(params.x[-n], model[-n:], params.theta[-n:])

        return loss

    def constraints(params: Parameters):
        y = time_march(train_x, model, params.theta)
        x_target = [train_x, *y[:-1]]

        x = np.hstack(params.x)
        x_target = np.hstack(x_target)

        defects = x - x_target
        return defects  # / inputs.shape[0]

    init_mult, lagrangian, get_param = fax.constrained.make_lagrangian(n_step_loss(1), constraints)
    initial_values = init_mult(Parameters(tuple(theta), tuple(x)))
    (theta, x), multiplier = fax.constrained.constrained_test.eg_solve(lagrangian, lambda *args: False, get_param, initial_values, max_iter=1000, metrics=None, lr=lr)

    push_metrics(train_x, train_y, model, theta, x, -1, full_rollout_loss)
    return theta


def train_accuracy(train_x, train_y, model, theta):
    target_class = np.argmax(train_y, axis=-1)
    y = time_march(train_x, model, theta)
    logits = y[-1]
    predicted_class = np.argmax(logits, axis=-1)
    return np.mean(predicted_class == target_class)


def push_metrics(train_x, train_y, model, theta, x, outer_iter, loss_fn):
    accuracy = train_accuracy(train_x, train_y, model, theta)
    train_loss = loss_fn(train_x, model, theta)
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
        with jax.disable_jit():
            main()
    except KeyboardInterrupt:
        plt.close("all")

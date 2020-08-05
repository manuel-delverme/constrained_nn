import jax
import jax.experimental.optimizers
import jax.experimental.stax as stax
import jax.lax
import jax.numpy as np
import jax.ops
import matplotlib.pyplot as plt

from main_fax import load_dataset


# import config

def block(out_dim):
    return stax.serial(
        # stax.Dense(32, ),
        # stax.LeakyRelu,
        stax.Dense(out_dim, ),
        stax.Softmax if out_dim == 1 else stax.LeakyRelu
    )


def make_net():
    return [
        block(32),
        # block(32),
        block(1),
    ]


def plot_model(xs, theta, trainX, trainY, title="trajectory"):
    fig, ax = plt.subplots()
    plt.title(title)
    ax.set_xlim(-0.1, len(xs) + 0.1)
    ax.set_ylim(-1.1, 5.1)
    ax.scatter(range(len(xs)), xs, c="cyan")
    ax.scatter((0, len(xs)), (trainX, trainY), c="black", s=15)
    x_t1s = time_march(trainX, theta)

    for t, (x_t, x_t1) in enumerate(zip([trainX, *x_t1s], x_t1s)):
        line, = ax.plot([t, t + 1], [x_t, x_t1], c="orange", linestyle="--", linewidth=2)
        line.set_label(f"simulation t={t}")

    network = make_net(theta)
    for t, (x, b) in enumerate(zip(xs, network)):
        line, = ax.plot([t, t + 1], [x, b(x)], c="green")
        line.set_label(f"1step t={t}")
    ax.legend()
    fig.show()


def time_march(x0, network, parameters):
    x_t1 = []
    x_t = x0
    for layer, theta_t in zip(network, parameters):
        x_t = layer(theta_t, x_t)
        x_t1.append(x_t)
    return x_t1


def trajectory_consistency(x, theta, trainX):
    return constraint(x, theta, trainX, 0)[:-1]


def constraint(x, theta, trainX, trainY):
    network = make_net(theta)
    defects = [
        np.array(x[0] - trainX).reshape(1, 1),
        x[1] - network[0](x[0]),
        trainY - network[-1](x[-1]),
    ]
    return defects


def main():
    num_outputs, train_x, train_y, test_x, test_y = load_dataset(normalize=True)
    train_x = np.ones_like(train_x)
    train_y = np.zeros_like(train_y)
    train_y = jax.ops.index_update(train_y, jax.ops.index[:, 0], 1)

    rng_key = jax.random.PRNGKey(0)
    blocks_init, blocks_predict = zip(*make_net())

    input_shape = train_x.shape
    batch_size = train_x.shape[0]
    parameters = []
    for init in blocks_init:
        _, init_params = init(rng_key, input_shape)
        parameters.append(init_params)
        input_shape = (batch_size, *init_params[-2][-1].shape)  # use bias shape since it's dense layers

    y = time_march(train_x, blocks_predict, parameters)
    x = [train_x, *y[:-1]]
    lr = 1e-4

    def accuracy(params, batch):
        inputs, targets = batch
        target_class = np.argmax(targets, axis=-1)
        x_ = time_march(inputs, blocks_predict, params)
        logits = x_[-1]
        predicted_class = np.argmax(logits, axis=-1)
        loss = -np.sum(logits * targets)
        return np.mean(predicted_class == target_class), loss

    for outer_iter in range(1000):
        x, network, losses = sequential(lr, blocks_predict, parameters, x, train_x, train_y, outer_iter)
        acc, loss = accuracy(network, (train_x, train_y))
        print(float(acc), float(loss), losses)


def sequential(lr, blocks_predict, theta, x, trainX, trainY, outer_iter):
    # y = [*x[1:], trainY]
    y = [*x[1:], None]
    losses = []

    for t in reversed(range(len(x))):
        theta_t = theta[t]
        block_t = blocks_predict[t]
        x_t = x[t]
        y_t = y[t]
        if t == 0:
            x_tm1 = None
            theta_tm1 = None
            block_tm1 = None
        else:
            x_tm1 = x[t - 1]
            theta_tm1 = theta[t - 1]
            block_tm1 = blocks_predict[t - 1]

        def loss(x_t, y_t, theta_t, x_tm1, theta_tm1):
            H = []
            if t != len(x) - 1:
                H.append(block_t(theta_t, x_t) - jax.lax.stop_gradient(y_t))

            if t >= 1 and theta_tm1 is not None:
                H.append(jax.lax.stop_gradient(block_tm1(theta_tm1, x_tm1)) - x_t, )
            elif t == 0:
                assert x_tm1 is None
                assert theta_tm1 is None
                H.append(x_t - trainX)

            return np.linalg.norm(np.hstack(H), 2)

        if t != len(x) - 1:
            theta[t] = theta_step(loss, lr, theta_t, x_t, y_t)
        x[t] = x_step(loss, lr, theta_t, x_t, y_t, x_tm1, theta_tm1)
        losses.append(float(loss(x_t, y_t, theta_t, x_tm1, theta_tm1)))

    return x, theta, losses


def theta_step(loss, lr, theta_t, x, y, num_steps=10):
    opt_init, opt_update, get_params = jax.experimental.optimizers.sgd(lr)

    @jax.jit
    def update(i, opt_state, x):
        theta_t = get_params(opt_state)
        param_grad = jax.grad(loss, 2)(x, y, theta_t, None, None)
        return opt_update(i, param_grad, opt_state)

    opt_state = opt_init(theta_t)
    for i in range(num_steps):
        opt_state = update(i, opt_state, x)
    theta_t = get_params(opt_state)

    return theta_t


def x_step(loss, lr, theta_t, x, y, x_tm1, theta_tm1, num_steps=10):
    opt_init, opt_update, get_params = jax.experimental.optimizers.sgd(lr)

    # opt_init, opt_update, get_params = jax.experimental.optimizers.momentum(lr, mass=0.9)

    @jax.jit
    def update(i, opt_state, x):
        x = get_params(opt_state)
        param_grad = jax.grad(loss, 0)(x, y, theta_t, x_tm1, theta_tm1)
        return opt_update(i, param_grad, opt_state)

    opt_state = opt_init(x)
    for i in range(num_steps):
        opt_state = update(i, opt_state, x)

    return get_params(opt_state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")

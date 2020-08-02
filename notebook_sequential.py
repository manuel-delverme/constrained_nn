import jax
import jax.lax
import jax.numpy as np
import jax.ops
import matplotlib.pyplot as plt
import numpy as onp


def make_net(theta):
    network = []

    def make_layer(theta):
        assert theta.shape == (1, 1)

        def f(x):
            return x + 10 * (jax.nn.sigmoid(theta) - 0.5)  # np.dot(theta, x)  # + b

        return f

    for theta_i in theta:
        network.append(make_layer(theta_i.reshape(-1, 1)))

    return network


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


def time_march(x0, theta):
    network = make_net(theta)
    Bx = []
    h = x0
    for layer in network:
        h = layer(h)
        Bx.append(h)
    return np.vstack(Bx)


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
    rng = onp.random.RandomState(0)
    size = (1, 1)
    theta = np.vstack([rng.normal(size=size) / 2 for _ in range(2)])
    trainX, trainY = 3., 1.
    x = np.zeros(theta.shape[0])
    x = jax.ops.index_update(x, 0, trainX)
    lr = 1e-2

    print('states:', x.T, '\nweights:', theta.T)
    print('shapes', x.shape, theta.shape)
    # twostep_gradient(lr, theta, trainX, trainY, x)

    plot_model(x, theta, trainX, trainY, "initial")
    for outer_iter in range(10):
        x, theta = sequential(lr, theta, trainX, trainY, x, outer_iter)
        # plot_model(x, theta, trainX, trainY, f"final_{outer_iter}")


def sequential(lr, theta, trainX, trainY, x, outer_iter):
    # TODO: consider x_t-2
    y = [*x[1:], trainY]
    # xm1 = [None, *x[1:]]

    # for t, (x_tm1, x_t, y_t, theta_t) in reversed(list(enumerate(zip(xm1, x, ys, theta)))):
    for t in reversed(range(len(x))):
        theta_t = theta[t]
        x_t = x[t]
        y_t = y[t]
        if t == 0:
            x_tm1 = None
            theta_tm1 = None
        else:
            x_tm1 = x[t - 1]
            theta_tm1 = theta[t - 1]

        def loss(x_t, y_t, theta_t, x_tm1, theta_tm1, norm=True):
            b, = make_net(theta_t)
            H = [
                b(x_t) - jax.lax.stop_gradient(y_t),
            ]
            if t >= 1 and theta_tm1 is not None:
                bm1, = make_net(theta_tm1)
                H.append(jax.lax.stop_gradient(bm1(x_tm1)) - x_t, )
            elif t == 0:
                assert x_tm1 is None
                assert theta_tm1 is None
                H.append(np.array(x_t - trainX).reshape((1, 1)))

            if norm:
                return np.linalg.norm(np.hstack(H), 2)
            else:
                return np.hstack(H)

        theta_t = theta_step(loss, lr, theta_t, x_t, y_t)
        theta = jax.ops.index_update(theta, t, theta_t.squeeze(-1))

        plot_model(x, theta, trainX, trainY, f"after theta_{t},{outer_iter}")
        print(loss(x_t, y_t, theta_t, None, None))

        x_t = x_step(loss, lr, theta_t, x_t, y_t, x_tm1, theta_tm1)
        x = jax.ops.index_update(x, t, x_t.squeeze(-1))

        plot_model(x, theta, trainX, trainY, f"after x_{t},{outer_iter}")
        print(loss(x_t, y_t, theta_t, x_tm1, theta_tm1))
    return x, theta


def theta_step(loss, lr, theta_t, x, y):
    Gph = jax.grad(loss, 2)
    for iter in range(100):
        param_grad = Gph(x, y, theta_t, None, None)
        theta_t += lr * -param_grad
    return theta_t


def x_step(loss, lr, theta_t, x, y, x_tm1, theta_tm1):
    Gph = jax.grad(loss, 0)
    for iter in range(100):
        state_grad = Gph(x, y, theta_t, x_tm1, theta_tm1)
        x += lr * -state_grad
    return x


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")

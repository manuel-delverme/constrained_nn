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
    # fig.show()
    plt.savefig(f"plots/{title}.png")


def time_march(x0, theta):
    network = make_net(theta)
    Bx = []
    h = x0
    for layer in network:
        h = layer(h)
        Bx.append(h)
    return np.vstack(Bx)


def constraint(x, theta, trainX, _trainY):
    network = make_net(theta)
    y_0 = network[0](x[0])
    defects = [
        np.array(x[0] - trainX).reshape(1, 1),
        x[1] - y_0,
    ]
    return np.hstack(defects)


def loss(x, theta, _trainX, trainY):
    network = make_net(theta)
    return trainY - network[-1](x[-1])


def main():
    rng = onp.random.RandomState(0)
    size = (1, 1)
    theta = np.vstack([rng.normal(size=size) / 4 for _ in range(2)])
    trainX, trainY = np.array(3.), np.array(1.)
    x = np.zeros(theta.shape[0])
    x = jax.ops.index_update(x, 0, trainX)

    print('states:', x.T, '\nweights:', theta.T)
    print('shapes', x.shape, theta.shape)

    # twostep_gradient(lr, theta, trainX, trainY, x)
    # plot_model(x, theta, trainX, trainY, "initial")

    def forwardprop_loss(theta):
        y = time_march(trainX, theta)
        predicted = y[-1]
        error = (trainY - predicted).reshape(1, )
        return np.linalg.norm(error, 2)

    # explicit_loss_gradient = jax.grad(forwardprop_loss)(theta)

    theta = theta.squeeze()
    for iter_num in range(100):
        gradient_implicit = implicit_diff(theta, trainX, trainY)
        theta = theta - 0.01 * gradient_implicit
        print(forwardprop_loss(theta))
        x_star = jax.lax.stop_gradient(find_root(theta, trainX))
        plot_model(x_star, theta, trainX, trainY, title=f"iter_{iter_num}")

    # print(explicit_loss_gradient, gradient_implicit)
    print()


def implicit_diff(theta, trainX, trainY):
    def rootfind_loss(theta, x):
        network = make_net(theta)
        predicted = network[-1](x[-1])
        error = (trainY - predicted).reshape(1, )
        return np.linalg.norm(error, 2)

    x_star = jax.lax.stop_gradient(find_root(theta, trainX))
    classification_loss = jax.grad(rootfind_loss, 0)(theta, x_star).squeeze()
    implicit_gradient = ift(theta, trainX, trainY, x_star)
    implicit_gradient = implicit_gradient[-1, :]
    gradient_implicit = classification_loss + implicit_gradient  # WHY ADD?
    return gradient_implicit


def find_root(theta, x0):
    y = time_march(x0, theta)
    x_star = np.append(x0, y[:-1])
    return x_star


def ift(theta, trainX, trainY, x):
    y = time_march(x[0], theta)
    x_star = np.append(x[0], y[:-1])
    # plot_model(x_star, theta, trainX, trainY)
    assert (np.array(constraint(x_star, theta, trainX, trainY)) == 0.).all()
    Dxh = jax.jacobian(constraint, 0)
    state_jac = Dxh(x, theta, trainX, trainY)
    state_jac = state_jac.squeeze()
    # print(state_jac.shape)
    # print(state_jac)
    Dph = jax.jacobian(constraint, 1)
    param_jac = Dph(x, theta, trainX, trainY)
    param_jac = param_jac.squeeze()
    # print(param_jac.shape)
    # print(param_jac)
    Dphi_p = -np.linalg.pinv(state_jac).dot(param_jac)

    # print(Dphi_x, Dphi_x.shape)
    # dx = np.dot(Dphi_x, state_jac)
    # dtheta = param_jac.squeeze[0]
    # https://timvieira.github.io/blog/post/2016/03/05/gradient-based-hyperparameter-optimization-and-the-implicit-function-theorem/
    return Dphi_p


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")

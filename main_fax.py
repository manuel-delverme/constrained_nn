import collections
import itertools
import time

import fax.competitive
import fax.constrained
import fax.constrained.constrained_test
import jax.numpy as jnp
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import numpy.random as npr
import tqdm
from jax import jit, grad, random
from jax.config import config
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax

import config
import datasets
import layers
import utils


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


init_random_params, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax)

if __name__ == "__main__":
    rng = random.PRNGKey(0)

    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)


    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]


    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)


    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)


    _, init_params = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {}".format(train_acc))
        print("Test set accuracy {}".format(test_acc))

raise NotImplementedError("done")

ConstrainedSolution = collections.namedtuple(
    "ConstrainedSolution",
    "value converged iterations"
)

convergence_params = dict(rtol=1e-7, atol=1e-7)


def load_dataset(normalize=False):
    dataset = config.dataset
    import sklearn.model_selection
    targets = dataset.target.reshape(-1)
    n_outputs = len(set(dataset.target))
    one_hot_targets = np.eye(n_outputs)[targets.astype(onp.int)]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.data, one_hot_targets, test_size=0.25, random_state=31337)
    trainX = X_train.astype(np.float32)
    trainY = y_train.astype(np.float32)
    testX = X_test.astype(np.float32)
    testY = y_test.astype(np.float32)
    if normalize:
        train_mean = np.mean(trainX, 0)
        train_std = np.std(trainX, 0)
        trainX -= train_mean
        trainX /= train_std
        testX -= train_mean
        testX /= train_std
    return n_outputs, trainX, trainY, testX, testY


def drawcurve(train_, valid_, id, legend_1, legend_2):
    acc_train = np.array(train_).flatten()
    acc_test = np.array(valid_).flatten()

    plt.figure(id)
    plt.semilogy(acc_train)
    plt.semilogy(acc_test)
    axes = plt.gca()
    axes.set_ylim([0, 1])

    plt.legend([legend_1, legend_2], loc='upper left')
    plt.show()


def make_block_nn(num_inputs, num_outputs, dataset_size) -> layers.BlockNN:
    model = [
        [layers.fc(num_inputs, config.num_hidden), ],
        [layers.fc(config.num_hidden, config.num_hidden), ],
        [layers.fc(config.num_hidden, num_outputs), ]
    ]
    blocks = []
    split_variables = []
    for i, block in enumerate(model):
        blocks.append(layers.NNBlock(block))
        var_out = blocks[-1].modules[-1].weights.shape[1]
        split_variables.append(onp.random.rand(dataset_size, var_out))
    del split_variables[-1]  # the last variable is y_target

    return layers.BlockNN(blocks, split_variables)


def run_experiment(num_outputs, trainX, trainY, testX, testY, iters):
    dataset_size, num_inputs = trainX.shape
    model = make_block_nn(num_inputs, num_outputs, dataset_size)

    indices = np.arange(trainX.shape[0])

    def sample_dataset(mini_batch):
        if mini_batch:
            mini_batch_indices = onp.random.choice(indices, config.batch_size, replace=False)
        else:
            mini_batch_indices = indices
        return mini_batch_indices

    def convergence_test(x_new, x_old):
        return False

    def train_accuracy(model, _multipliers):
        predicted = model(trainX)
        accuracy = np.argmax(trainY, axis=1) == np.argmax(predicted, axis=1)
        return accuracy.mean()

    def objective_function(model, mini_batch=config.batch_size > 0):
        mini_batch_indices = sample_dataset(mini_batch)
        batch_x, batch_y = trainX[mini_batch_indices, :], trainY[mini_batch_indices, :]
        loss = model.loss(batch_x, batch_y, mini_batch_indices)
        return -loss

    def equality_constraints(model, mini_batch=config.batch_size):
        mini_batch_indices = sample_dataset(mini_batch)
        batchX = trainX[mini_batch_indices, :]
        return model.constraints(batchX, mini_batch_indices)

    init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(objective_function, equality_constraints)
    initial_values = init_mult(model)

    print('iters', iters)
    metrics = [
        ("train/objective_function", lambda model, l: objective_function(model, mini_batch=False).mean()),
        ("train/equality_constraints", lambda model, l: equality_constraints(model, mini_batch=False).mean()),
        ("train/loss", lagrangian),
        ("train/train_accuracy", train_accuracy)
    ]
    if config.use_sgd:
        x, multiplier, history = utils.sgd_solve(lagrangian, convergence_test, get_x, initial_values, iters, metrics, config.lr)
    else:
        x, multiplier, history = fax.constrained.constrained_test.eg_solve(lagrangian, convergence_test, get_x, initial_values, iters, metrics, config.lr)
    return dict(history)


def plot_learning_curves(curves):
    for name, values in curves.items():
        values = np.array(values)
        _, axes = plt.subplots(1, 1, figsize=(20, 5))
        axes.set_xlabel("Training examples")
        axes.set_title(name)

        train_scores_mean = np.mean(values, axis=0)
        train_scores_std = np.std(values, axis=0)
        train_sizes = np.arange(values.shape[1])

        axes.grid()
        axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
        axes.plot(train_sizes, train_scores_mean, label=name)  # 'o-', color="r",
        axes.legend(loc="best")
        plt.show()


def main():
    num_outputs, train_x, train_y, test_x, test_y = load_dataset(normalize=True)
    logs = collections.defaultdict(list)

    for _ in tqdm.trange(config.num_experiments):
        history = run_experiment(num_outputs, train_x, train_y, test_x, test_y, iters=config.optimization_iters)
        for k, v in history.items():
            logs[k].append(v)

    plot_learning_curves(logs)


if __name__ == "__main__":
    main()

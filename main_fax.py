import collections

import fax.constrained
import fax.constrained.constrained_test
# import fax.utils
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import tqdm

import config
import layers
import utils

ConstrainedSolution = collections.namedtuple(
    "ConstrainedSolution",
    "value converged iterations"
)

convergence_params = dict(rtol=1e-7, atol=1e-7)


def load_dataset(normalize=True):
    dataset = config.dataset
    import sklearn.model_selection
    targets = dataset.target.reshape(-1)
    n_outputs = len(set(dataset.target))
    one_hot_targets = np.eye(n_outputs)[targets.astype(onp.int)]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.data, one_hot_targets, test_size=0.25, random_state=31337)
    trainX = X_train.astype(np.float64)
    trainY = y_train.astype(np.float64)
    testX = X_test.astype(np.float64)
    testY = y_test.astype(np.float64)
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

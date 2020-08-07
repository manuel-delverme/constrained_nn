import collections

import fax.loop
import jax
import jax.numpy as np
import numpy as onp
import tqdm

import config
import layers
from main_fax import load_dataset, plot_learning_curves  # , make_block_nn
from utils import plot_model


def make_block_nn(num_inputs, num_outputs, dataset_size) -> layers.BlockNN:
    model = [
        [
            layers.toy_fc(),
        ], [
            layers.toy_fc(),
        ]
    ]
    blocks = []
    split_variables = []
    for i, block in enumerate(model):
        blocks.append(layers.NNBlock(block))
        var_out = blocks[-1].modules[-1].weights.shape[1]
        split_variables.append(np.full((dataset_size, var_out), 2.))
    del split_variables[-1]  # the last variable is y_target

    return layers.BlockNN(layers.NNParameters(blocks), np.array(split_variables))


def run_experiment(num_outputs, trainX, trainY, testX, testY):
    def train_accuracy(model, _multipliers):
        predicted = model(trainX)
        accuracy = np.argmax(trainY, axis=1) == np.argmax(predicted, axis=1)
        return accuracy.mean()

    onp.random.seed(2)

    dataset_size, num_inputs = trainX.shape
    model = make_block_nn(num_inputs, num_outputs, dataset_size)

    def convergence_test(x_new, x_old):
        return False

    history = collections.defaultdict(list)

    i = 0
    plot_model(model, trainX, trainY, "init", i)

    for opt_step in range(config.optimization_iters):
        with jax.disable_jit():
            theta_err, model = h_step(trainX, trainY, convergence_test, config.optimization_subiters, model)
        i += 1
        plot_model(model, trainX, trainY, "theta", i)
        print("theta_err", theta_err)

        x_err, model = x_step(trainX, trainY, convergence_test, config.optimization_subiters, model)
        i += 1
        plot_model(model, trainX, trainY, "x", i)
        print("x_err", x_err)
        # push_metrics(history, model, opt_step, theta_err, train_accuracy, x_err)

    return model, dict(history)


def x_step(x0, xT, convergence_test, iters, model):
    states = [x0, *model.split_variables, xT]

    def objective_function(_var):
        # random_projection = onp.random.rand(x_t.shape[1], block.modules[0].weights.shape[1])
        # random_projection /= random_projection.sum(axis=0)
        # random_x_t = np.dot(x_t, random_projection)
        # return np.linalg.norm(x_hat - inputs, 2) / inputs.shape[0]
        return np.array(0.)

    def equality_constraints(split_variables):
        states_ = np.vstack([states[0], *split_variables, states[-1]])
        defects = []
        for b, x, y in zip(model.blocks, states_[:-1], states_[1:]):
            defect = b(x) - y
            defects.append(defect)

        error = np.linalg.norm(np.vstack(defects), 2)
        num = np.array(len(defect))
        retr = error / num
        return retr

    init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(objective_function, equality_constraints)
    initial_values = init_mult(model.split_variables)

    x, multiplier = fax.constrained.constrained_test.eg_solve(lagrangian, convergence_test, get_x, initial_values, iters, None, config.lr)
    error = equality_constraints(x)
    m = layers.BlockNN(model.blocks, x)

    return np.mean(np.array(error)), m


def h_step(x0, xT, convergence_test, iters, model):
    states = np.vstack([x0, *model.split_variables, xT])

    def objective_function(_var):
        # random_projection = onp.random.rand(x_t.shape[1], block.modules[0].weights.shape[1])
        # random_projection /= random_projection.sum(axis=0)
        # random_x_t = np.dot(x_t, random_projection)
        # return np.linalg.norm(x_hat - inputs, 2) / inputs.shape[0]
        return 0

    def equality_constraints(blocks_):
        defects = []
        # h = jax.vmap(blocks_, 0)
        # print(1)
        for b, x, y in zip(blocks_, states[:-1], states[1:]):
            defect = b(x) - y
            defects.append(defect)

        error = np.linalg.norm(np.vstack(defects), 2)
        # num = np.array(len(defect))
        return error  # / num

    init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(objective_function, equality_constraints)
    initial_values = init_mult(model.blocks)

    x, multiplier = fax.constrained.constrained_test.eg_solve(lagrangian, convergence_test, get_x, initial_values, iters, None, config.lr)
    error = equality_constraints(x)
    m = layers.BlockNN(x, model.split_variables)
    del model
    return np.mean(np.array(error)), m


def main():
    num_outputs, train_x, train_y, test_x, test_y = load_dataset(normalize=True)
    logs = collections.defaultdict(list)

    train_x = np.full((1, 1), 0, dtype=np.float32)
    train_y = np.full((1, 1), 1, dtype=np.float32)

    for _ in tqdm.trange(config.num_experiments):
        model, history = run_experiment(num_outputs, train_x, train_y, test_x, test_y)
        for k, v in history.items():
            logs[k].append(v)

    plot_learning_curves(logs)


if __name__ == "__main__":
    main()

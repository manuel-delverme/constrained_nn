import collections

import fax.loop
import jax.numpy as np
import numpy as onp
import tqdm

import config
import layers
from main_fax import load_dataset, plot_learning_curves  # , make_block_nn


def make_block_nn(num_inputs, num_outputs, dataset_size) -> layers.BlockNN:
    model = [
        [
            layers.fc(num_inputs, config.num_hidden),
            # layers.fc(config.num_hidden, config.num_hidden),
            # ], [
            #     layers.fc(config.num_hidden, config.num_hidden),
            #     # layers.fc(config.num_hidden, config.num_hidden),
        ], [
            # layers.fc(config.num_hidden, config.num_hidden),
            layers.fc(config.num_hidden, num_outputs),
        ]
    ]
    blocks = []
    split_variables = []
    for i, block in enumerate(model):
        blocks.append(layers.NNBlock(block))
        var_out = blocks[-1].modules[-1].weights.shape[1]
        split_variables.append(onp.random.rand(dataset_size, var_out))
    del split_variables[-1]  # the last variable is y_target

    return layers.BlockNN(blocks, split_variables)


def run_experiment(num_outputs, trainX, trainY, testX, testY):
    # dataset_size, num_inputs = trainX.shape
    # model = make_block_nn(num_inputs, num_outputs, dataset_size)
    def train_accuracy(model, _multipliers):
        predicted = model(trainX)
        accuracy = np.argmax(trainY, axis=1) == np.argmax(predicted, axis=1)
        return accuracy.mean()

    onp.random.seed(2)

    dataset_size, num_inputs = trainX.shape
    model = make_block_nn(num_inputs, num_outputs, dataset_size)

    block_outs = [*model.split_variables, trainY]
    block_ins = [trainX, *model.split_variables]

    def convergence_test(x_new, x_old):
        return False

    history = collections.defaultdict(list)

    for opt_step in range(config.optimization_iters):
        theta_err = h_step(block_ins, block_outs, convergence_test, config.optimization_subiters, model)
        print("theta_err", theta_err)
        x_err = x_step(block_ins, block_outs, convergence_test, config.optimization_subiters, model)
        print("x_err", x_err)

        accuracy = train_accuracy(model, None)
        metrics = [
            ("train/x_err", x_err),
            ("train/theta_err", theta_err),
            ("train/train_accuracy", accuracy)
        ]
        for tag, value in metrics:
            config.tb.add_scalar(tag, float(value), opt_step)
            history[tag].append(value)

    return model, dict(history)


def x_step(block_ins, block_outs, convergence_test, iters, model):
    error = []
    for idx, (block, x_t, y_t) in reversed(list(enumerate(zip(model.blocks, block_ins, block_outs)))):
        if idx == 0:
            # x0 is not a split variable
            continue

        def objective_function(_var):
            # random_projection = onp.random.rand(x_t.shape[1], block.modules[0].weights.shape[1])
            # random_projection /= random_projection.sum(axis=0)
            # random_x_t = np.dot(x_t, random_projection)
            # return np.linalg.norm(x_hat - inputs, 2) / inputs.shape[0]
            return 0

        def equality_constraints(_x_t):
            defect = block(_x_t) - y_t
            error = np.linalg.norm(defect, 2)
            return error / x_t.shape[0]

        init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(objective_function, equality_constraints)
        initial_values = init_mult(x_t)

        x, multiplier = fax.constrained.constrained_test.eg_solve(lagrangian, convergence_test, get_x, initial_values, iters, None, config.lr)
        error.append(equality_constraints(x))
        assert model.split_variables[idx - 1].shape == x.shape
        model.split_variables[idx - 1] = x
    return np.mean(np.array(error))


def h_step(block_ins, block_outs, convergence_test, iters, model):
    error = []
    for idx, (block, x_t, y_t) in reversed(list(enumerate(zip(model.blocks, block_ins, block_outs)))):
        assert (x_t.shape[1], y_t.shape[1]) == block.modules[0].weights.shape

        def objective_function(_var):
            # random_projection = onp.random.rand(x_t.shape[1], block.modules[0].weights.shape[1])
            # random_projection /= random_projection.sum(axis=0)
            # random_x_t = np.dot(x_t, random_projection)
            # return np.linalg.norm(x_hat - inputs, 2) / inputs.shape[0]
            return 0

        def equality_constraints(block_):
            defect = block_(x_t) - y_t
            error = np.linalg.norm(defect, 2)
            return error / x_t.shape[0]

        init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(objective_function, equality_constraints)
        initial_values = init_mult(block)

        x, multiplier = fax.constrained.constrained_test.eg_solve(lagrangian, convergence_test, get_x, initial_values, iters, None, config.lr)
        error.append(equality_constraints(x))
        assert x.modules[0].weights.shape == block.modules[0].weights.shape
        model.blocks[idx] = x
    return np.mean(np.array(error))


def main():
    num_outputs, train_x, train_y, test_x, test_y = load_dataset(normalize=True)
    logs = collections.defaultdict(list)

    for _ in tqdm.trange(config.num_experiments):
        model, history = run_experiment(num_outputs, train_x, train_y, test_x, test_y)
        for k, v in history.items():
            logs[k].append(v)

    plot_learning_curves(logs)


if __name__ == "__main__":
    main()

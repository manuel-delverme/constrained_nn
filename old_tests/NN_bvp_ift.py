import collections

import fax.loop
import jax
import jax.experimental.optimizers
import jax.numpy as np
import numpy as onp
import tqdm

import config
import layers
from main_fax import load_dataset  # , make_block_nn


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


def sgd_solve(function, convergence_test, get_x, initial_values, max_iter=100000000, lr=None):
    optimizer_init, optimizer_update, optimizer_get_params = jax.experimental.optimizers.adam(lr)

    @jax.jit
    def update(i, opt_state):
        x = optimizer_get_params(opt_state)
        grads = jax.grad(function)(x)
        return optimizer_update(i, grads, opt_state)

    solution = fax.loop.fixed_point_iteration(
        init_x=optimizer_init(initial_values),
        func=update,
        convergence_test=convergence_test,
        max_iter=max_iter,
        get_params=optimizer_get_params,
    )
    return get_x(solution)


def run_experiment(num_outputs, trainX, trainY, testX, testY, iters):
    # dataset_size, num_inputs = trainX.shape
    # model = make_block_nn(num_inputs, num_outputs, dataset_size)
    onp.random.seed(2)

    dataset_size, num_inputs = trainX.shape
    model = make_block_nn(num_inputs, num_outputs, dataset_size)

    block_outs = [*model.split_variables, trainY]
    block_ins = [trainX, *model.split_variables]

    def convergence_test(x_new, x_old):
        return False

    def get_x(x):
        return x

    def train_accuracy(model, _multipliers):
        predicted = model(trainX)
        accuracy = np.argmax(trainY, axis=1) == np.argmax(predicted, axis=1)
        return accuracy.mean()

    for idx, (b, x_t, y_t) in reversed(list(enumerate(zip(model.blocks, block_ins, block_outs)))):
        # def function(ins):
        #     error = b(ins) - y_t
        #     return np.mean(np.square(error))

        def objective_function(block, x_t):
            random_projection = onp.random.rand(x_t.shape[1], block.modules[0].weights.shape[1])
            random_projection /= random_projection.sum(axis=0)
            random_x_t = np.dot(x_t, random_projection)
            return np.linalg.norm(x_hat - inputs, 2) / inputs.shape[0]

        def equality_constraints(block, x_t):
            defect = block(x_t) - y_t
            return np.linalg.norm(defect, 2)  # / ins.shape[0]

        init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(objective_function, equality_constraints)
        initial_values = init_mult(b)
        x, multiplier, history = fax.constrained.constrained_test.eg_solve(lagrangian, convergence_test, get_x, initial_values, iters, None, config.lr)

        print(function(x.value))
        model.split_variables[idx - 1] = x.value
    return model


def main():
    num_outputs, train_x, train_y, test_x, test_y = load_dataset(normalize=True)
    logs = collections.defaultdict(list)

    for _ in tqdm.trange(1):
        model = run_experiment(num_outputs, train_x, train_y, test_x, test_y, iters=10000)
        print(model)


if __name__ == "__main__":
    main()

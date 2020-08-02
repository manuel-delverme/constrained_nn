import collections

import jax
import jax.numpy as np
import numpy as onp
import tqdm

import layers
from main_fax import load_dataset  # , make_block_nn

np.convolve


def find_roots(function, x0, tol=1.48e-8, maxiter=50, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
    return


def make_block_nn(num_inputs, num_outputs, dataset_size) -> layers.BlockNN:
    num_hidden = 32
    model = [
        [layers.fc(num_inputs, num_hidden), ],
        [layers.fc(num_hidden, num_hidden), ],
        [layers.fc(num_hidden, num_outputs), ]
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
    onp.random.seed(2)
    trainX = onp.random.rand(10, 2)
    trainY = onp.random.rand(10, num_outputs)

    dataset_size, num_inputs = trainX.shape
    model = make_block_nn(num_inputs, num_outputs, dataset_size)

    forward(model, trainX)
    backward(model, values)


def forward(model, trainX):
    for _ in range(2):
        block_outs = [*model.split_variables]
        block_ins = [trainX, *model.split_variables[:-1]]
        for idx, (b, bin, bout) in enumerate(zip(model.blocks[:-1], block_ins, block_outs)):
            block_out = b(bin)
            print(np.linalg.norm(block_out - bout, 2))
            model.split_variables[idx] = block_out


def backward(model, trainX):
    X, THETA = 0, 1
    h = model.constraints
    theta = model

    M1 = jax.grad(h, THETA)(0, theta)
    M2 = jax.grad(h, THETA)(0, theta)
    W1 = np.linalg.inv(M1)
    dx_dtheta = - W1 * M2

    # dl_dx = ...
    dl_dtheta = dl_dx * dx_dtheta + dl_dtheta

    for _ in range(2):
        block_outs = [*model.split_variables]
        block_ins = [trainX, *model.split_variables[:-1]]
        for idx, (b, bin, bout) in enumerate(zip(model.blocks[:-1], block_ins, block_outs)):
            block_out = b(bin)
            print(np.linalg.norm(block_out - bout, 2))
            model.split_variables[idx] = block_out


def main():
    num_outputs, train_x, train_y, test_x, test_y = load_dataset(normalize=True)
    logs = collections.defaultdict(list)

    for _ in tqdm.trange(1):
        history = run_experiment(num_outputs, train_x, train_y, test_x, test_y, iters=3)
        for k, v in history.items():
            logs[k].append(v)


if __name__ == "__main__":
    main()

# replace twophase forward solver
# replace forward solver by newton
# replace forward solver

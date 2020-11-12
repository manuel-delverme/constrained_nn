# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.experimental.stax is for neural network building, and
the mini-library jax.experimental.optimizers is for first-order stochastic
optimization.
"""

import itertools
import time

import jax
import jax.numpy as np
import numpy.random as npr
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax

import config
import datasets


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -np.mean(np.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


def update_metrics(params, iter_num, train_images, train_labels, test_images, test_labels):
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    config.tb.add_scalar('train/accuracy', float(train_acc), iter_num)
    config.tb.add_scalar('test/accuracy', float(test_acc), iter_num)


depth = sum(config.blocks)

init_random_params, predict = stax.serial(
    *[Dense(1024), Relu, ] * (depth - 1),
    Dense(10), LogSoftmax
)

if __name__ == "__main__":
    rng = random.PRNGKey(0)

    step_size = 0.001
    num_epochs = config.num_epochs
    batch_size = config.batch_size
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
    next_eval = 0
    rng_key = jax.random.PRNGKey(0)

    print("\nStarting training...")
    for iter_num in range(num_epochs):
        start_time = time.time()

        opt_state = update(iter_num, opt_state, next(batches))

        if next_eval == iter_num:
            params = get_params(opt_state)
            update_metrics(params, iter_num, train_images, train_labels, test_images, test_labels)
            rng_key, k_out = jax.random.split(rng_key)
            next_eval += int(config.eval_every + jax.random.randint(k_out, (1,), 0, config.eval_every // 100))

        epoch_time = time.time() - start_time


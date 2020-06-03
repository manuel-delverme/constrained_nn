import collections

import jax
import jax.experimental.stax
import jax.numpy as np
import numpy as onp
from jax import tree_util


class FC(collections.namedtuple("FC", "weights bias")):
    def __call__(self, inputs):
        y = np.dot(inputs, self.weights) + self.bias
        return y


class NNBlock(collections.namedtuple("model", "modules", )):
    def __call__(self, inputs):
        h = inputs
        for module in self.modules:
            pre_h = module(h)
            h = jax.nn.leaky_relu(pre_h)
        y_hat = h
        return y_hat


class BlockNN(collections.namedtuple("BlockNN", "blocks split_variables", )):
    def loss(self, inputs, outputs, mini_batch_indices):
        last_block = self.blocks[-1]
        y_hat = last_block(self.split_variables[-1][mini_batch_indices])
        return np.linalg.norm(y_hat - outputs, 2) / outputs.shape[0]

    def constraints(self, inputs, samples_indices):
        constraints = []
        split_variables_batch = [var[samples_indices] for var in self.split_variables]

        splits_left = [inputs, *split_variables_batch[:-1]]

        for a, block, h in zip(splits_left, self.blocks, split_variables_batch):
            # jax.lax.stop_gradient() ?
            constraints.append(h - block(a))
        return np.hstack(constraints)  # / inputs.shape[0]

    def __call__(self, inputs):
        hidden_state = inputs
        for block in self.blocks:
            hidden_state = block(hidden_state)
        y_hat = hidden_state
        return y_hat

    def __sub__(self, other):
        return tree_util.tree_multimap(lambda _a, _b: _a - _b, self, other)

    def __add__(self, other):
        return tree_util.tree_multimap(lambda _a, _b: _a + _b, self, other)


def fc(num_inputs, num_outputs):
    init_fn, apply_fn = jax.experimental.stax.Dense(num_outputs)
    return FC(onp.random.rand(num_inputs, num_outputs), onp.random.rand(1, num_outputs))

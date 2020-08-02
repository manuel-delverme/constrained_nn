import collections

# import fax.utils
import jax.numpy as np
import numpy as onp
from jax import tree_util
from jax.nn import leaky_relu
from jax.nn import softmax


class FC(collections.namedtuple("FC", "weights bias")):
    def __call__(self, inputs):
        y = np.dot(inputs, self.weights) + self.bias
        return y


class NNBlock(collections.namedtuple("model", "modules", )):
    def __call__(self, inputs):
        h = inputs
        for module in self.modules:
            pre_h = module(h)
            h = leaky_relu(pre_h)
        y_hat = h
        return y_hat

    def __sub__(self, other):
        return tree_util.tree_multimap(lambda _a, _b: _a - _b, self, other)

    def __add__(self, other):
        return tree_util.tree_multimap(lambda _a, _b: _a + _b, self, other)


class NNParameters(collections.namedtuple("NNParams", "params", )):
    def __sub__(self, other):
        return tree_util.tree_multimap(lambda _a, _b: _a - _b, self, other)

    def __add__(self, other):
        return tree_util.tree_multimap(lambda _a, _b: _a + _b, self, other)

    def __iter__(self):
        return self.params.__iter__()


class BlockNN(collections.namedtuple("BlockNN", "blocks split_variables", )):
    def loss(self, inputs, outputs, mini_batch_indices):
        last_block = self.blocks[-1]
        y_hat = last_block(self.split_variables[-1][mini_batch_indices])
        return np.linalg.norm(y_hat - outputs, 2) / outputs.shape[0]

    def constraints(self, inputs, samples_indices):
        constraints = 0.
        split_variables_batch = [var[samples_indices] for var in self.split_variables]
        splits_left = [inputs, *split_variables_batch[:-1]]

        # this could be replaced by vmap(np.linalg.norm(split_variables_batch - vmap(block(split_left)))
        input_batch_arguments = 0, None  # Batch first dimension
        # vmap(function, input_batch_arguments, output_batch_arguments)
        for a, block, h in zip(splits_left, self.blocks, split_variables_batch):
            # jax.lax.stop_gradient() ?
            constraints += np.linalg.norm(h - block(a), 2) / h.shape[0]
        return constraints  # / inputs.shape[0]

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


def fc(num_inputs, num_outputs, bias=True):
    if bias:
        return FC(onp.random.rand(num_inputs, num_outputs), onp.random.rand(1, num_outputs))
    return FC(onp.random.rand(num_inputs, num_outputs), np.zeros(1))


def toy_fc():
    class toyFC(collections.namedtuple("toyFC", "weights")):
        def __call__(self, inputs):
            # y = inputs + np.clip(self.weights, -.75, .75)
            y = inputs + softmax(self.weights)
            return y

    return toyFC(np.full((1, 1), 1e-1))

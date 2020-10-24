from jax.experimental import stax as stax

import config


def make_block_net(num_outputs):
    network = []
    for t, block_size in enumerate(config.blocks):
        block = []
        for ti in range(block_size):
            block.append(stax.Dense(config.num_hidden, ), )
            block.append(stax.LeakyRelu)
            # block.append(stax.Relu)
        network.append(block)

    network[-1][-2] = stax.Dense(num_outputs, )
    network[-1][-1] = stax.LogSoftmax
    network = [stax.serial(*n) for n in network]
    return zip(*network)

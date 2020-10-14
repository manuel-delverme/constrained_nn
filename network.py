from jax.experimental import stax as stax

import config


def make_block_net(num_outputs):
    network = []
    for t, block_size in enumerate(config.blocks):
        block = []
        for ti in range(block_size):
            if t == len(config.blocks) - 1 and ti == block_size - 1:
                block.append(stax.Dense(num_outputs, ), )
                block.append(stax.LogSoftmax)
            else:
                block.append(stax.Dense(config.num_hidden, ), )
                block.append(stax.LeakyRelu)
        network.append(stax.serial(*block))

    return zip(*network)

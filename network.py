from jax.experimental import stax as stax

import config


def make_block_net(num_outputs):
    return zip(*[
        stax.serial(
            # stax.Dense(1024, ),
            stax.Dense(config.num_hidden, ),
            stax.LeakyRelu,
        ),
        stax.serial(
            stax.Dense(config.num_hidden, ),
            stax.LeakyRelu,
        ),
        stax.serial(
            stax.Dense(num_outputs),
            stax.Softmax
            # stax.LogSoftmax
        ),
    ])

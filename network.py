from jax.experimental import stax as stax


def make_block_net(num_outputs):
    return zip(*[
        stax.serial(
            # stax.Dense(1024, ),
            stax.Dense(32, ),
            stax.LeakyRelu,
        ),
        stax.serial(
            # stax.Dense(1024, ),
            stax.Dense(32, ),
            stax.LeakyRelu,
        ),
        stax.serial(
            stax.Dense(num_outputs),
            stax.Softmax
            # stax.LogSoftmax
        ),
    ])

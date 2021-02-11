import itertools

from jax import jit, grad
from jax.experimental import optimizers

import train
import utils


def main():
    step_size = 0.1
    num_epochs = 100
    momentum_mass = 0.0

    batch_gen, model, initial_parameters, full_batch, num_batches = train.initialize()
    full_rollout_loss, last_layer_loss, equality_constraints = train.make_losses(model)

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        g = grad(last_layer_loss, 0)(params, batch)

        return opt_update(i, g, opt_state)

    opt_state = opt_init(initial_parameters)
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batch_gen))

        params = get_params(opt_state)
        train_loss_, train_acc_ = utils.train_acc(full_batch.x, full_batch.y, model, params.theta)
        print(f"Training set accuracy {train_acc_}")
        print(f"Training set loss {train_loss_}")
    return train_acc_ == 1.


if __name__ == "__main__":
    main()
    # with jax.disable_jit():
    #     main()

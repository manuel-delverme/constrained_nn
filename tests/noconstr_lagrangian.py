import itertools

from jax import jit, grad

import stacked_FC_parallel
import utils

num_epochs = 100


def main():
    full_batch, model, opt_state, optimizer_get_params, lagrangian, optimizer_update, batch_gen, num_batches = stacked_FC_parallel.init_opt_problem()

    @jit
    def update(i, opt_state, batch):
        # g = grad(last_layer_loss, 0, )
        g = grad(lagrangian, 0, )
        return optimizer_update(i, g, opt_state, batch)

    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batch_gen))

        params = optimizer_get_params(opt_state)
        train_loss_, train_acc_ = utils.train_acc(full_batch.x, full_batch.y, model, params.constr_params.theta)
        print(f"Training set accuracy {train_acc_}")
        print(f"Training set loss {train_loss_}")
    return train_acc_ == 1.


if __name__ == "__main__":
    main()
    # with jax.disable_jit():
    #     main()

import collections

import jax
import jax.numpy as np
import numpy as onp
import tqdm

import layers
from main_fax import load_dataset  # , make_block_nn


def newtons_method(function, x0, tol=1.48e-8, maxiter=50, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True):
    fprime = jax.jacrev(function)

    if tol <= 0:
        raise ValueError(f"tol too small ({tol:g} <= 0)")

    # Explicitly copy `x0` as `p` will be modified inplace, but, the
    # user's array should not be altered.
    p = np.array(x0, copy=True, dtype=float)

    failures = np.ones_like(p, dtype=bool)
    nz_der = np.ones_like(failures)

    # Newton-Raphson method
    for iteration in range(maxiter):
        # first evaluateecc fval
        fval = function(p)
        if (fval == 0).all():
            # all roots have been found, then terminate
            failures = fval.astype(bool)
            break
        jacobian = fprime(p)
        nz_der = (jacobian != 0)
        # stop iterating if all derivatives are zero
        if not nz_der.any():
            break
        # Newton step
        # jacobian = jacobian[nz_der]
        inv_jac = jax.vmap(jax.numpy.linalg.inv(jacobian))
        dp = inv_jac * fval[nz_der]
        if fprime2 is not None:
            fder2 = np.asarray(fprime2(p, *args))
            dp = dp / (1.0 - 0.5 * dp * fder2[nz_der] / jacobian[nz_der])
        # only update nonzero derivatives
        p[nz_der] -= dp
        failures[nz_der] = np.abs(dp) >= tol  # items not yet converged
        # stop iterating if there aren't any failures, not incl zero der
        if not failures[nz_der].any():
            break

    zero_der = ~nz_der & failures  # don't include converged with zero-ders
    if zero_der.any():
        # Secant warnings
        if fprime is None:
            nonzero_dp = (p1 != p)
            # non-zero dp, but infinite newton step
            zero_der_nz_dp = (zero_der & nonzero_dp)
            if zero_der_nz_dp.any():
                rms = np.sqrt(
                    sum((p1[zero_der_nz_dp] - p[zero_der_nz_dp]) ** 2)
                )
                warnings.warn(f'RMS of {rms:g} reached', RuntimeWarning)
        # Newton or Halley warnings
        else:
            all_or_some = 'all' if zero_der.all() else 'some'
            msg = f'{all_or_some:s} derivatives were zero'
            warnings.warn(msg, RuntimeWarning)
    elif failures.any():
        all_or_some = 'all' if failures.all() else 'some'
        msg = f'{all_or_some:s} failed to converge after {maxiter:d} iterations'
        if failures.all():
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning)

    if full_output:
        result = namedtuple('result', ('root', 'converged', 'zero_der'))
        p = result(p, ~failures, zero_der)

    return p
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
    trainY = onp.random.rand(10, 2)

    dataset_size, num_inputs = trainX.shape
    model = make_block_nn(num_inputs, num_outputs, dataset_size)

    block_outs = [*model.split_variables, trainY]
    block_ins = [trainX, *model.split_variables]

    new_splits = []

    for b, bin, bout in zip(model.blocks, block_ins, block_outs):
        def function(ins):
            defects = b(ins) - bout
            # return np.mean(np.square(error))
            return defects

        block_out = newtons_method(function, bin)
        new_splits.append(block_out)
    print(new_splits)


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

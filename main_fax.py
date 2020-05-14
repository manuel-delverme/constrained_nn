import collections

import fax.constrained
import fax.constrained.constrained_test
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import tqdm

import config

convergence_params = dict(rtol=1e-7, atol=1e-7)


def load_dataset():
    dataset = config.dataset
    import sklearn.model_selection
    targets = dataset.target.reshape(-1)
    n_outputs = len(set(dataset.target))
    one_hot_targets = np.eye(n_outputs)[targets.astype(onp.int)]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.data, one_hot_targets, test_size=0.25, random_state=31337)
    trainX = X_train.astype(np.float32)
    trainY = y_train.astype(np.float32)
    testX = X_test.astype(np.float32)
    testY = y_test.astype(np.float32)
    return n_outputs, trainX, trainY, testX, testY


def drawcurve(train_, valid_, id, legend_1, legend_2):
    acc_train = np.array(train_).flatten()
    acc_test = np.array(valid_).flatten()

    plt.figure(id)
    plt.semilogy(acc_train)
    plt.semilogy(acc_test)
    axes = plt.gca()
    axes.set_ylim([0, 1])

    plt.legend([legend_1, legend_2], loc='upper left')
    plt.show()


def run_experiment(n_outputs, trainX, trainY, testX, testY):
    n_batches, num_features = trainX.shape
    train_epochs = 15
    warm_epochs = 5
    Model = collections.namedtuple("model", ("w1", "h1", "w2", "h2", "w3"), )
    model = Model(
        np.zeros((num_features, config.num_hidden)),  # w1
        onp.random.rand(n_batches, config.num_hidden),  # h1
        onp.random.rand(config.num_hidden, config.num_hidden_last),  # w2
        onp.random.rand(n_batches, config.num_hidden_last),  # h2
        onp.random.rand(config.num_hidden_last, n_outputs),  # w3
    )

    def convergence_test(x_new, x_old):
        return fax.converge.max_diff_test(x_new, x_old, **convergence_params)

    def _relu(x):
        return np.maximum(x, 0)

    def feed_forward(model, inputs):
        a1 = np.dot(inputs, model.w1)
        h1 = _relu(a1)
        # a2 = np.dot(h1, model.w2)
        # h2 = _relu(a2)
        y_hat = np.dot(h1, model.w3)
        return y_hat

    def objective_function(model):
        predicted = feed_forward(model, trainX)
        # accuracy = np.argmax(testY, axis=1) == np.argmax(predicted, axis=1)
        loss = np.mean(np.power(predicted - trainY, 2))
        return -loss

    def train_accuracy(*args):
        if len(args) == 1:
            model = args
        else:
            model, _ = args
        predicted = feed_forward(model, trainX)
        accuracy = np.argmax(trainY, axis=1) == np.argmax(predicted, axis=1)
        # accuracy = np.argmax(testY, axis=1) == np.argmax(predicted, axis=1)
        return accuracy.mean()

    def equality_constraints(model):
        h = [
            model.h1 - np.dot(trainX, model.w1),
            model.h2 - np.dot(model.h1, model.w2),
        ]
        return np.hstack(h)

    init_mult, lagrangian, get_x = fax.constrained.make_lagrangian(objective_function, equality_constraints)
    initial_values = init_mult(model)

    for it in range(3, 5):
        final_val, h, x, multiplier = fax.constrained.constrained_test.eg_solve(
            lagrangian, convergence_test, equality_constraints, objective_function, get_x, initial_values, max_iter=10 ** (1 + it), metrics=[lagrangian, train_accuracy])
        print(train_accuracy(model))

    list_loss_train = []
    list_loss_valid = []
    list_accuracy_train = []
    list_accuracy_valid = []

    def objective_function(model):
        predicted = feed_forward(model, trainX)
        # accuracy = np.argmax(testY, axis=1) == np.argmax(predicted, axis=1)
        loss = np.mean(np.power(predicted - trainY, 2))
        return -loss

    for i in range(train_epochs):
        loss_train, accuracy_train = model.fit(trainX, trainY, beta=1.0, gamma=10.0, update_lagrangian=i > warm_epochs, fit_elephant=i == train_epochs - 1)
        loss_valid, accuracy_valid = model.evaluate(testX, testY)

        # Append loss and accuracy
        list_loss_train.append(loss_train)
        list_loss_valid.append(loss_valid)
        list_accuracy_train.append(accuracy_train)
        list_accuracy_valid.append(accuracy_valid)

        # if i > 2 and i % 20== 0:
        #     plot_learning_curve(np.stack([list_accuracy_train]), np.stack([list_accuracy_valid]))

    print(list_accuracy_valid[-1])
    # plot_learning_curve(np.stack([list_accuracy_train]), np.stack([list_accuracy_valid]))
    return list_accuracy_train, list_accuracy_valid


def plot_learning_curve(train_scores, test_scores):
    _, axes = plt.subplots(1, 1, figsize=(20, 5))
    # axes.set_ylim(0.0, 1)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=0)
    train_scores_std = np.std(train_scores, axis=0)
    test_scores_mean = np.mean(test_scores, axis=0)
    test_scores_std = np.std(test_scores, axis=0)
    train_sizes = np.arange(train_scores.shape[1])

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Train")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test")
    axes.legend(loc="best")
    plt.show()


def main():
    print("With four parameters I can fit an elephant, and with five I can make him wiggle his trunk")
    print("With thousands i hope to fit the exploration dilemma (and an elephant)")

    args = load_dataset()
    tas, vas = [], []

    for _ in tqdm.trange(1):
        ta, va = run_experiment(*args)
        tas.append(ta)
        vas.append(va)

    # drawcurve(list_accuracy_train, list_accuracy_valid, 2, 'acc_train', 'acc_valid')
    plot_learning_curve(np.stack(tas), np.stack(vas))


if __name__ == "__main__":
    main()

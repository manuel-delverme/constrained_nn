import matplotlib.pyplot as plt
import numpy as np
import tqdm

import ADMM_NN


def load_dataset():
    import sklearn.datasets
    iris = sklearn.datasets.load_breast_cancer()
    # .load_iris()

    # from tensorflow.examples.tutorials.mnist import input_data
    # tfe.enable_eager_execution()
    # Load MNIST data
    # mnist = input_data.read_data_sets("./data/", one_hot=True)
    import sklearn.model_selection
    targets = iris.target.reshape(-1)
    n_outputs = len(set(iris.target))
    one_hot_targets = np.eye(n_outputs)[targets]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(iris.data, one_hot_targets, test_size=0.25, random_state=31337)
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

    n_hiddens = 32  # number of neurons

    train_epochs = 100
    warm_epochs = 10

    beta = 1.0  # weight of activation to latent
    gamma = 10.0  # weight of latent to output
    model = ADMM_NN.ADMM_NN(num_features, n_hiddens, n_outputs, n_batches)
    model.warm_up(trainX, trainY, warm_epochs, beta, gamma)
    list_loss_train = []
    list_loss_valid = []
    list_accuracy_train = []
    list_accuracy_valid = []
    for i in tqdm.trange(train_epochs):
        # print("------ Training: {:d} ------".format(i))
        loss_train, accuracy_train = model.fit(trainX, trainY, beta, gamma)
        loss_valid, accuracy_valid = model.evaluate(testX, testY)

        # tqdm.tqdm.write(f"Loss train: {np.array(loss_train):3f}, accuracy train: {np.array(accuracy_train):3f}")
        # tqdm.tqdm.write(f"Loss valid: {np.array(loss_valid):3f}, accuracy valid: {np.array(accuracy_valid):3f}")

        # Append loss and accuracy
        list_loss_train.append(loss_train)
        list_loss_valid.append(loss_valid)
        list_accuracy_train.append(accuracy_train)
        list_accuracy_valid.append(accuracy_valid)

    print(list_accuracy_valid[-1])
    return list_accuracy_train, list_accuracy_valid


def plot_learning_curve(train_scores, test_scores):
    _, axes = plt.subplots(1, 1, figsize=(20, 5))
    axes.set_ylim(0.7, 1)
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

    for _ in range(10):
        ta, va = run_experiment(*args)
        tas.append(ta)
        vas.append(va)

    # drawcurve(list_accuracy_train, list_accuracy_valid, 2, 'acc_train', 'acc_valid')
    plot_learning_curve(np.stack(tas), np.stack(vas))


if __name__ == "__main__":
    main()

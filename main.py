import matplotlib.pyplot as plt
import numpy as np
import numpy as onp
import tqdm

import ADMM_NN
import config


def load_dataset():
    dataset = config.dataset
    import sklearn.model_selection
    targets = dataset.target.reshape(-1)
    n_outputs = len(set(dataset.target))
    one_hot_targets = np.eye(n_outputs)[targets.astype(onp.int)]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.data, one_hot_targets, test_size=0.60, random_state=31337)
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

    train_epochs = 30
    warm_epochs = 10

    model = ADMM_NN.ADMM_NN(num_features, n_outputs, n_batches)
    # model.warm_up(trainX, trainY, warm_epochs, beta, gamma)
    list_loss_train = []
    list_loss_valid = []
    list_accuracy_train = []
    list_accuracy_valid = []
    for i in tqdm.trange(train_epochs):
        # print("------ Training: {:d} ------".format(i))
        loss_train, accuracy_train = model.fit(trainX, trainY, beta=1.0, gamma=10.0, update_lagrangian=i > warm_epochs, fit_elephant=i == train_epochs - 1)
        # loss_train, accuracy_train = model.fit(trainX, trainY, beta, gamma)
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


def main():
    print("With four parameters I can fit an elephant, and with five I can make him wiggle his trunk")
    print("With thousands i hope to fit the exploration dilemma (and an elephant)")

    args = load_dataset()
    tas, vas = [], []

    for _ in tqdm.trange(10):
        ta, va = run_experiment(*args)
        tas.append(ta)
        vas.append(va)
        # drawcurve(list_accuracy_train, list_accuracy_valid, 2, 'acc_train', 'acc_valid')


if __name__ == "__main__":
    main()

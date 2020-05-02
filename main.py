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
    trainX = np.transpose(X_train).astype(np.float32)
    trainY = np.transpose(y_train).astype(np.float32)
    testX = np.transpose(X_test).astype(np.float32)
    testY = np.transpose(y_test).astype(np.float32)
    return n_outputs, trainX, trainY, testX, testY


print("With four parameters I can fit an elephant, and with five I can make him wiggle his trunk")
print("With thousands i hope to fit the exploration dilemma (and an elephant)")

n_outputs, trainX, trainY, testX, testY = load_dataset()
num_features, n_batches = trainX.shape

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

    # if i and (i % (train_epochs // 10)) == 0:
    #     # Drawing loss, accuracy of train and valid
    #     model.drawcurve(list_loss_train, list_loss_valid, 1, 'loss_train', 'loss_valid')
    #     model.drawcurve(list_accuracy_train, list_accuracy_valid, 2, 'acc_train', 'acc_valid')

# Drawing loss, accuracy of train and valid
print(list_accuracy_valid[-1])
# model.drawcurve(list_loss_train, list_loss_valid, 1, 'loss_train', 'loss_valid')
model.drawcurve(list_accuracy_train, list_accuracy_valid, 2, 'acc_train', 'acc_valid')

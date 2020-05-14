import sklearn.datasets
import tensorboardX

weighted_payload = False
attack = False
# dataset = sklearn.datasets.fetch_openml('mnist_784')
dataset = sklearn.datasets.load_iris()
num_hidden = 32
num_hidden_last = 32

tb = tensorboardX.SummaryWriter()

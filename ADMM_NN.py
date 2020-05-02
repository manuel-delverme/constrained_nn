from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tqdm


class ADMM_NN:
    """ Class for ADMM Neural Network. """

    def __init__(self, n_inputs, n_hiddens, n_outputs, n_batches):

        """
        Initialize variables for NN.
        Raises:
            ValueError: Column input samples, for example, the input size of MNIST data should be (28x28, *) instead of (*, 28x28).
        :param n_inputs: Number of inputs.
        :param n_hiddens: Number of hidden units.
        :param n_outputs: Number of outputs
        :param n_batches: Number of data sample that you want to train
        :param return:
        """
        self.a0 = np.zeros((n_batches, n_inputs))

        self.w1 = np.zeros((n_inputs, n_hiddens))  # None
        self.w2 = np.random.rand(n_hiddens, n_hiddens)
        self.w3 = np.random.rand(n_hiddens, n_outputs)

        self.z1 = np.random.rand(n_batches, n_hiddens)
        self.a1 = np.random.rand(n_batches, n_hiddens)

        self.z2 = np.random.rand(n_batches, n_hiddens)
        self.a2 = np.random.rand(n_batches, n_hiddens)

        self.z3 = np.random.rand(n_batches, n_outputs)

        self.lambda_lagrange = np.ones((n_batches, n_outputs))

    @staticmethod
    def _weight_update(z, a):
        """
        Consider it now the minimization of the problem with respect to W_l.
        For each layer l, the optimal solution minimizes ||z_l - W_l a_l-1||^2. This is simply
        a least square problem, and the solution is given by W_l = z_l p_l-1, where p_l-1
        represents the pseudo-inverse of the rectangular activation matrix a_l-1.
        :param z: output matrix (z_l)
        :param a: activation matrix l-1  (a_l-1)
        :return: weight matrix
        """
        pinv = np.linalg.pinv(a)
        # print(np.linalg.norm(a.dot(pinv)))
        w = np.dot(pinv, z)

        def null_space(a):
            aap = np.linalg.pinv(a).dot(a)
            return np.eye(*aap.shape) - aap

        # def null(A, eps=1e-15):
        #    u, s, vh = np.linalg.svd(A)
        #    return np.transpose(vh)

        null_proj = null_space(a)
        # trace = a.dot(null_proj)
        # print(np.linalg.norm(trace))

        # rnd = np.random.rand(a.shape[1], z.shape[1])
        payload = np.random.rand(null_proj.shape[1], w.shape[1])

        def meh_max(x):
            meh_x = x - np.min(x)
            return meh_x / meh_x.sum(axis=0)
        weights = meh_max(1 / np.abs(a.dot(null_proj)).mean(0))
        payload = np.einsum('xy,x->xy', payload, weights)

        w_new = w - null_proj.dot(payload)

        # print(np.linalg.norm(z - a.dot(w)))
        # w_new = np.dot(pinv, z) - null_proj.dot(payload)
        # print(np.linalg.norm(z - a.dot(w_new)))

        # proj_c = np.dot(null_proj, rnd)
        # print(np.dot(w, proj_c.T))

        return w_new

    @staticmethod
    def _activation_update(next_weight, next_layer_output, layer_nl_output, beta, gamma):
        """
        Minimization for a_l is a simple least squares problem similar to the weight update.
        However, in this case the matrix appears in two penalty terms in the problem, and so
        we must minimize:
            beta ||z_l+1 - W_l+1 a_l||^2 + gamma ||a_l - h(z_l)||^2
        :param next_weight:  weight matrix l+1 (w_l+1)
        :param next_layer_output: output matrix l+1 (z_l+1)
        :param layer_nl_output: activate output matrix h(z) (h(z_l))
        :param beta: value of beta
        :param gamma: value of gamma
        :return: activation matrix
        """
        # Calculate ReLU
        layer_nl_output = _relu(layer_nl_output)

        # Activation inverse
        m1 = beta * np.dot(next_weight, next_weight.T)

        m2 = gamma * np.eye(m1.shape[0])
        av = np.linalg.inv(m1 + m2)

        # Activation formulate
        m3 = beta * np.dot(next_layer_output, next_weight.T)
        m4 = gamma * layer_nl_output
        af = m3 + m4

        # Output
        return np.dot(af, av)

    def _argminz(self, a_outputs, layer_weight, a_inputs, beta_weight_cost, gamma_activation_cost):
        """
        This problem is non-convex and non-quadratic (because of the non-linear term h).
        Fortunately, because the non-linearity h works entry-wise on its argument, the entries
        in z_l are decoupled. This is particularly easy when h is piecewise linear, as it can
        be solved in closed form; common piecewise linear choices for h include rectified
        linear units (ReLUs), that its used here, and non-differentiable sigmoid functions.
        :param a_outputs: activation matrix (a_l)
        :param layer_weight:  weight matrix (w_l)
        :param a_inputs: activation matrix l-1 (a_l-1)
        :param beta_weight_cost: value of beta
        :param gamma_activation_cost: value of gamma
        :return: output matrix
        """
        a_hat = np.dot(a_inputs, layer_weight)
        sol1 = (gamma_activation_cost * a_outputs + beta_weight_cost * a_hat) / (gamma_activation_cost + beta_weight_cost)
        sol2 = a_hat
        z1 = np.zeros_like(a_outputs)
        z2 = np.zeros_like(a_outputs)
        z = np.zeros_like(a_outputs)

        # sol1 = np.array(sol1)
        # sol2 = np.array(sol2)

        z1[sol1 >= 0.] = sol1[sol1 >= 0.]
        z2[sol2 <= 0.] = sol2[sol2 <= 0.]

        fz_1 = np.square(gamma_activation_cost * (a_outputs - _relu(z1))) + beta_weight_cost * (np.square(z1 - a_hat))
        fz_2 = np.square(gamma_activation_cost * (a_outputs - _relu(z2))) + beta_weight_cost * (np.square(z2 - a_hat))

        # fz_1 = np.array(fz_1)
        # fz_2 = np.array(fz_2)

        index_z1 = fz_1 <= fz_2
        index_z2 = fz_1 > fz_2

        z[index_z1] = z1[index_z1]
        z[index_z2] = z2[index_z2]

        return z

    def _argminlastz(self, targets, eps, w, a_in, beta):
        """
        Minimization of the last output matrix, using the above function.
        :param targets: target matrix (equal dimensions of z) (y)
        :param eps: lagrange multiplier matrix (equal dimensions of z) (lambda)
        :param w: weight matrix (w_l)
        :param a_in: activation matrix l-1 (a_l-1)
        :param beta: value of beta
        :return: output matrix last layer
        """
        m = np.dot(a_in, w)
        z = (targets - eps + beta * m) / (1 + beta)
        return z

    def _lambda_update(self, zl, w, a_in, beta):
        """
        Lagrange multiplier update.
        :param zl: output matrix last layer (z_L)
        :param w: weight matrix last layer (w_L)
        :param a_in: activation matrix l-1 (a_L-1)
        :param beta: value of beta
        :return: lagrange update
        """
        mpt = np.dot(a_in, w)
        lambda_up = beta * (zl - mpt)
        return lambda_up

    def feed_forward(self, inputs):
        outputs = _relu(np.dot(inputs, self.w1))
        outputs = _relu(np.dot(outputs, self.w2))
        outputs = np.dot(outputs, self.w3)
        return outputs

    def fit(self, inputs, labels, beta, gamma):
        # Input layer
        self.w1 = self._weight_update(self.z1, inputs)
        self.a1 = self._activation_update(self.w2, self.z2, self.z1, beta, gamma)
        self.z1 = self._argminz(self.a1, self.w1, self.a0, beta, gamma)

        # Hidden layer
        self.w2 = self._weight_update(self.z2, self.a1)
        self.a2 = self._activation_update(self.w3, self.z3, self.z2, beta, gamma)
        self.z2 = self._argminz(self.a2, self.w2, self.a1, beta, gamma)

        # Output layer
        self.w3 = self._weight_update(self.z3, self.a2)
        self.z3 = self._argminlastz(labels, self.lambda_lagrange, self.w3, self.a2, beta)
        self.lambda_lagrange += self._lambda_update(self.z3, self.w3, self.a2, beta)

        loss, accuracy = self.evaluate(inputs, labels)

        return loss, accuracy

    def evaluate(self, inputs, labels, isCategrories=True):
        predicted = self.feed_forward(inputs)
        loss = np.mean(np.abs(predicted - labels))

        if isCategrories:
            accuracy = np.argmax(labels, axis=1) == np.argmax(predicted, axis=1)
            accuracy = accuracy.mean()

        else:
            accuracy = loss

        return loss, accuracy

    def warm_up(self, inputs, labels, epochs, beta, gamma):
        """
        Warming ADMM Neural Network by minimizing sub-problems without update lambda
        :param inputs: input of training data samples
        :param outputs: label of training data samples
        :param epochs: number of epochs
        :param beta: value of beta
        :param gamma: value of gamma
        :return:
        """
        for _ in tqdm.trange(epochs):
            # Input layer

            self.w1 = self._weight_update(self.z1, inputs)
            self.a1 = self._activation_update(self.w2, self.z2, self.z1, beta, gamma)
            self.z1 = self._argminz(self.a1, self.w1, self.a0, beta, gamma)

            # # Hidden layer
            self.w2 = self._weight_update(self.z2, self.a1)
            self.a2 = self._activation_update(self.w3, self.z3, self.z2, beta, gamma)
            self.z2 = self._argminz(self.a2, self.w2, self.a1, beta, gamma)

            # Output layer
            self.w3 = self._weight_update(self.z3, self.a2)
            self.z3 = self._argminlastz(labels, self.lambda_lagrange, self.w3, self.a2, beta)


def _relu(x):
    """
    Relu activation function
    :param x: input x
    :return: max 0 and x
    """
    return np.maximum(x, 0)

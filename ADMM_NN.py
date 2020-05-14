from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import config


class ADMM_NN:
    """ Class for ADMM Neural Network. """

    def __init__(self, n_inputs, n_outputs, n_batches):

        self.a0 = np.zeros((n_batches, n_inputs))

        self.w1 = np.zeros((n_inputs, config.num_hidden))  # None
        self.w2 = np.random.rand(config.num_hidden, config.num_hidden_last)
        self.w3 = np.random.rand(config.num_hidden_last, n_outputs)

        self.z1 = np.random.rand(n_batches, config.num_hidden)
        self.a1 = np.random.rand(n_batches, config.num_hidden)

        self.z2 = np.random.rand(n_batches, config.num_hidden_last)
        self.a2 = np.random.rand(n_batches, config.num_hidden)

        self.z3 = np.random.rand(n_batches, n_outputs)

        self.lambda_lagrange = np.ones((n_batches, n_outputs))

    @staticmethod
    def _weight_update(z, a, fit_elephant=False):
        pinv = np.linalg.pinv(a, rcond=1e-15)
        # print(np.linalg.norm(a.dot(pinv)))
        w = np.dot(pinv, z)

        if config.attack and fit_elephant:
            def null_space(a):
                aap = np.linalg.pinv(a).dot(a)
                return np.eye(*aap.shape) - aap

            # def null(A, eps=1e-15):
            #    u, s, vh = np.linalg.svd(A)
            #    return np.transpose(vh)

            null_proj = null_space(a)
            # trace = a.dot(null_proj)
            # print(np.linalg.norm(trace))

            elephant = np.random.rand(a.shape[0], z.shape[1]) * 1e16
            payload = np.dot(pinv, elephant)

            if config.weighted_payload:
                def meh_max(x):
                    meh_x = x - np.min(x)
                    return meh_x / meh_x.sum(axis=0)

                weights = meh_max(1 / np.abs(a.dot(null_proj)).mean(0))
                payload = np.einsum('xy,x->xy', payload, weights)

            correction = null_proj.dot(payload)
            print('elephant size', np.linalg.norm(correction))
            w_new = w - correction
        else:
            w_new = w

        # print(np.linalg.norm(z - a.dot(w)))
        # w_new = np.dot(pinv, z) - null_proj.dot(payload)
        # print(np.linalg.norm(z - a.dot(w_new)))

        # proj_c = np.dot(null_proj, rnd)
        # print(np.dot(w, proj_c.T))

        return w_new

    @staticmethod
    def _activation_update(next_weight, next_layer_output, layer_nl_output, beta, gamma):
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

        z1[sol1 >= 0.] = sol1[sol1 >= 0.]
        z2[sol2 <= 0.] = sol2[sol2 <= 0.]

        fz_1 = np.square(gamma_activation_cost * (a_outputs - _relu(z1))) + beta_weight_cost * (np.square(z1 - a_hat))
        fz_2 = np.square(gamma_activation_cost * (a_outputs - _relu(z2))) + beta_weight_cost * (np.square(z2 - a_hat))

        index_z1 = fz_1 <= fz_2
        index_z2 = fz_1 > fz_2

        z[index_z1] = z1[index_z1]
        z[index_z2] = z2[index_z2]

        return z

    def _lambda_update(self, zl, w, a_in, beta):
        mpt = np.dot(a_in, w)
        lambda_up = beta * (zl - mpt)
        return lambda_up

    def feed_forward(self, inputs):
        outputs = _relu(np.dot(inputs, self.w1))
        outputs = _relu(np.dot(outputs, self.w2))
        outputs = np.dot(outputs, self.w3)
        return outputs

    def fit(self, inputs, labels, beta, gamma, update_lagrangian=True, fit_elephant=False):
        # Input layer
        self.w1 = self._weight_update(self.z1, inputs)
        self.a1 = self._activation_update(self.w2, self.z2, self.z1, beta, gamma)
        self.z1 = self._argminz(self.a1, self.w1, self.a0, beta, gamma)

        # Hidden layer
        self.w2 = self._weight_update(self.z2, self.a1)
        self.a2 = self._activation_update(self.w3, self.z3, self.z2, beta, gamma)
        self.z2 = self._argminz(self.a2, self.w2, self.a1, beta, gamma)

        # Output layer
        self.w3 = self._weight_update(self.z3, self.a2, fit_elephant=fit_elephant)

        z3_hat = np.dot(self.a2, self.w3)
        self.z3 = (labels - self.lambda_lagrange + beta * z3_hat) / (1 + beta)

        if update_lagrangian:
            self.lambda_lagrange += self._lambda_update(self.z3, self.w3, self.a2, beta)

        return self.evaluate(inputs, labels)

    def evaluate(self, inputs, labels, isCategrories=True):
        predicted = self.feed_forward(inputs)
        loss = np.mean(np.abs(predicted - labels))

        if isCategrories:
            accuracy = np.argmax(labels, axis=1) == np.argmax(predicted, axis=1)
            accuracy = accuracy.mean()

        else:
            accuracy = loss

        return loss, accuracy


def _relu(x):
    """
    Relu activation function
    :param x: input x
    :return: max 0 and x
    """
    return np.maximum(x, 0)

from abc import ABC, abstractmethod


import numpy as np
import tensorflow as tf


class Activation(ABC):
    """
    activation function for each Node
    """
    def __init__(self, *args, **kwargs):
        """
        initialize instance properties
        """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        run aactivation function
        """


class Relu(Activation):
    """
    Rectified Linear Unit
    """
    @staticmethod
    def __call__(z):
        def r_max(vector):
            return np.where(vector >= 0, vector, 0.0)
        rslt = np.apply_along_axis(
            r_max,
            1,
            z
        )
        return rslt


class DRelu(Activation):
    """
    first deravitive of Relu
    """
    @staticmethod
    @np.vectorize
    def __call__(vector):
        if vector >= 0:
            return 1
        return 0

class Tanh(Activation):
    def __call__(self):
        pass


class DTanh(Activation):
    def __call__(self):
        pass


class Sigmoid(Activation):
    def __call__(self):
        pass


class DSigmoid(Activation):
    def __call__(self):
        pass


class Softmax(Activation):
    def __call__(self, z):
        np.exp(z, out=z)
        sum_ = np.sum(z, axis=1)[:, np.newaxis]
        return z / sum_


class DSoftmax(Activation):
    def __init__(self):
        self.delta = 0.001
        self.softmax = Softmax()

    def __call__(self, z):
        z = z.astype(np.float64)
        z_hat = z.copy()
        z_hat += self.delta
        rslt = (self.softmax(z_hat) - self.softmax(z)) / self.delta
        return rslt


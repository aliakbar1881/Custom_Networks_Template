from abc import ABC, abstractmethod
from collections import defaultdict


import numpy as np


class Optimizer(ABC):
    """
    optimizer for fastest convergence
    """
    def __init__(self, *args, **kwargs):
        """
        initialize instance properties
        """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        run Optimizer
        """


class SGD(Optimizer):
    def __init__(self, alpha, limit=0.001):
        self.alpha = alpha

    def __call__(self, deravitive, layer):
        layer.weights -= (self.alpha * deravitive * layer.weights)
        layer.bias -= self.alpha * deravitive


class MomentSGD(Optimizer):
    def __init__(self, beta):
        self.beta = beta
        self.v = defaultdict(lambda: (0.0, 0.0))

    def __call__(self, deravitative, layer):
        v0 = self.beta * self.v[layer.name][0] + (1 - self.beta) * deravitative[0]
        v1 = self.beta * self.v[layer.name][1] + (1 - self.beta) * deravitative[1]
        layer.weights -= v0
        layer.bias -= v1
        self.v[layer.name] = (np.sum(v0)/v0.shape[0], np.sum(v1)/v1.shape[0])

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['v']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.v = defaultdict(lambda:(0.0, 0.0))


class RMSPROP(Optimizer):
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate

    def __call__(self):
        pass


class AdaGard(Optimizer):
    def __init__(self):
        pass

    def __call__(self):
        pass


class Adam(Optimizer):
    def __init__(self, decay_rate, alpha):
        pass

    def __call__(self):
        pass

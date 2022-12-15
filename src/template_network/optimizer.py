from typing import namedtuples
from collections import defaultdict


from base import optimizer


class SGD(Optimizer):
    def __init__(self, alpha, limit=0.001):
        self.alpha = alpha

    def __call__(self, deravitive, layer):
        layer.weights -= (self.alpha * deravitive * layer.weights)
        layer.bias -= self.alpha * deravitive

class MomentSGD(Optimizer):
    def __init__(self, beta):
        self.beta = beta
        self.v = defaultdict(lambda: (0, 0))

    def __call__(self, deravitive, layer):
        v0 = self.beta * self.v[layer.name][0] + (1 - self.beta) * deravitive
        v1 = self.beta * self.v[layer.name][1] + (1 - self.beta) * deravitive
        layer.weights -= v0
        layer.bias -= v1
        self.v['layer.name'] = (v0, v1)


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

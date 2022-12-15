"""
various types of netwrok layer implememnt in this file
"""
from abc import ABC, abstractmethod

import numpy as np


import activation


class Layer(ABC):
    """
    base class of any layer in network
    """
    activations = {
        'relu': (activation.Relu, activation.DRelu),
        'tanh': (activation.Tanh, activation.DTanh),
        'sigmoid': (activation.Sigmoid, activation.DSigmoid),
        'softmax': (activation.Softmax, activation.DSoftmax),
    }

    def __init__(self, *args, **kwargs):
        """
        initialize instance properties
        """
        self.weights = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        run layer
        """


class FullyConnect(Layer):
    """
    fully connected dense layer that use perceptrons
    """
    def __init__(self, nodes, activation="relu"):
        super().__init__()
        """
        constructor of instance
        """
        try:
            self.activation, self.d_activation = self.activations[activation]
        except KeyError:
            if callable(activation):
                self.activation = activation
            else:
                raise f'{activation} is not activation function'
        self.nodes = nodes
        self.input_shape = None
        self.bias = None

    def _set_weights(self):
        self.weights = np.random.normal(
            0,
            1/self.nodes,
            (self.nodes, self.input_shape)
        )
        self.bias = np.random.uniform(self.nodes)

    def __call__(self, input_shape):
        self.input_shape = input_shape
        return self.nodes

    def forward_prop(self, input_) -> tuple:
        z = np.apply_along_axis(
            lambda i: self.weights @ i + self.bias,
            0,
            input_
        )
        output = tuple(
            z,
            np.apply_along_axis(self.d_activation, 0, z),
            np.apply_along_axis(self.activation, 0, z)
        )
        return output


if __name__ == '__main__':
    fully_layer = FullyConnect(10)
    fully_layer(32)
    fully_layer._set_weights()

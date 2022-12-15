from abc import ABC, abstractmethod


import numpy as np


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
    @np.vectorize
    def __call__(vector):
        if vector >= 0:
            return vector
        return 0


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

from abc import abstractmethod, ABC
import pickle


import numpy as np


class Model(ABC):
    """
    network architecture
    """
    def __init__(self, *args, **kwargs):
        """
        initialize instance properties
        """

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        initialize instance properties
        """

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model


class MLP(Model):
    def __init__(self, layers: list, input_shape: int):
        self.layers = layers
        self.input_shape = input_shape
        x = input_shape
        for i in self.layers:
            x = i(x)
            i._set_weights()
        self.optimizer = None
        self.loss = None
        self.train_loss = []
        self.dev_loss = []
        self.data = None
        self.labels = None
        self.deravitives = []

    def fit(self, data, labels, validation, epochs=1):
        self.data = data
        self.labels = labels
        for _ in range(epochs):
            y_hats = self.epoch()
            self.back_prop(y_hats, validation)
        return self.train_loss, self.dev_loss

    def back_prop(self, y_hats, validation):
        e = self.labels - y_hats[-1][-1]/self.labels.shape[0]
        der = (e, e)
        num = len(self.layers)
        for layer in reversed(self.layers):
            w = np.sum(
              np.sum(der[0]) / len(der[0])
              * np.sum(y_hats[num][1], axis=0)
              / y_hats[num][1].shape[0],
              axis=0
            )
            b = np.sum(
              np.sum(der[1]) / len(der[1])
              * np.sum(y_hats[num][1], axis=0)
              / y_hats[num][1].shape[0],
              axis=0
            )
            der = (w * layer.weights, np.full(layer.bias.shape[0], b))
            self.update(der, layer)
            num -= 1
        self.dev_loss.append(self.evaluate(*validation))
        self.train_loss.append(self.loss(y_hats[-1][-1], self.labels))
        print('train_loss: ', self.train_loss[-1])
        print('dev_loss: ', self.dev_loss[-1])

    def epoch(self):
        y_hats = [(self.data,)]
        for i in self.layers:
            y_hats.append(i.forward_prop(y_hats[-1][-1]))
        return y_hats

    def update(self, der, layer):
        self.optimizer(der, layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def evaluate(self, dev, label):
        y_hat = (None, dev)
        for i in self.layers:
            y_hat = i.forward_prop(y_hat[-1])
        return self.loss(label, y_hat[-1])

    def predict(self, input_):
        y_hat = (input_[np.newaxis, :],)
        for i in self.layers:
            y_hat = i.forward_prop(y_hat[-1])
        return y_hat[-1]

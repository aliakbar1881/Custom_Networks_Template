from abc import abstractmethod, ABC


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
        self.losses = []
        self.data = None
        self.labels = None
        self.deravitives = []

    def fit(self, data, labels, epochs=1):
        self.data = data
        self.labels = labels
        for _ in epochs:
            y_hats = self.epoch()
            self.back_prop(y_hats)

    def back_prop(self, y_hats):
        der = [self.labels - y_hats[-1]]
        for j, layer in enumerate(reversed(self.layers)):
            for i in range(0, j+1, -1):
                der *= y_hats[i][1] * y_hats[i][2] * layer.weights
                if i == j:
                    break
            self.update(der, layer.weights)
        self.losses.append(self.loss(y_hats, self.labels))
        print(self.losses[-1])

    def epoch(self):
        y_hats = [self.data]
        for i in self.layers:
            y_hats.append(i.forward_prop(y_hats[-1]))
        return y_hats

    def update(self, der, weights):
        self.optimizer(der, weights)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

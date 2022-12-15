class Loss(ABC):
    """
    loss function to caluclate differences between perdiction and actual value
    """
    def __init__(self, *args, **kwargs):
        """
        initiailize instance properties
        """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        run Loss function
        """


class MeanSquaredError(Loss):
    def __call__(self, y_hat, label):
        return label - y_hat

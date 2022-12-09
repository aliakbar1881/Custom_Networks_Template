"""
base classes to override in any specific situation
"""
from abc import abstractmethod, ABC


class Model(ABC):
    """
    network architecture
    """
    def __init__(self, *args, **kwargs):
        """
        initialize instance properties
        """

    def __call__(self, *args, **kwargs):
        """
        initialize instance properties
        """


class Layer(ABC):
    """
    base class of any layer in network
    """
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

#      @abstractmethod
    #  def _input(self, *args, **kwargs):
    #      """
    #      input fetaure
    #      """
    #
    #  @abstractmethod
    #  def _output(self, *args, **kwargs):
    #      """
    #      output
    #      """


class Node(ABC):
    """
    any node in each layer
    """
    def __init__(self, *args, **kwargs):
        """
        initialize instance properties
        """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        run node
        """


class Activation(ABC):
    """
    activation function for each Node
    """
    def __init__(self, *args, **kwargs):
        """
        initialize instance properties
        """

    def __call__(self, *args, **kwargs):
        """
        run aactivation function
        """


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


class Loss(ABC):
    """
    loss function to caluclate differences between perdiction and actual value
    """
    def __init__(self, *args, **kwargs):
        """
        initiailize instance properties
        """

    def __call__(self, *args, **kwargs):
        """
        run Loss function
        """

"""
base classes to override in any specific situation
"""



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




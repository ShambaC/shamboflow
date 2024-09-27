"""Base class for layers"""

class BaseLayer :
    """The base class for layers to inherit from.

    A layer is an object that is a part of
    the neural network. It performs calculations
    on neurons or neurons present in it
    commonly. It can also not have computable
    neurons, instead can also be used to filter
    data from previous layers.

    Attributes
    ----------
        name :
            The name of the layer
        trainable :
            Is the layer trainable
        is_built :
            Is the layer built and initialized
    """

    def __init__(self, name : str, trainable : bool = True) -> None:
        """Constructor for the base layer class

        Args
        ----
            name : str
                Name of the layer
            trainable : bool
                Is the layer trainable. Default is True
        
        """
        self.name = name
        self.trainable = trainable
        self.is_built = False

    def build(self) -> None :
        """Method to build and initialize the layer.

        This method is to be overidden by child
        classes to build them by their own logic.
        
        """
        self.is_built = True

    def compute(self) -> None :
        """Method to compute various parameter
        
        This method is to be implemented in inherited classes
        """
        pass
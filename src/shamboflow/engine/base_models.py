"""Base class for models"""

from shamboflow.engine.base_layers import BaseLayer

class BaseModel :
    """The Base class for models
    
    All other models need to
    inherit from this class.
    It provides a template with
    all the methods a model will
    need to execute.

    Attributes :
    ------------
        layers :
            A list of layers present in the model in the order from input to output
        weights :
            A list of all weight matrices for the layers in the model
        loss :
            The loss/cost function to use calculate error of the model at a given state
        train_data_x :
            The training data or features
        train_data_y :
            Training features' corresponding labels
        and more

        
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the required attributes.

        Also optionally takes in a list of
        layers as a parameter.

        `layers` is an optional arg.
        It takes a list of layers and them to the model
        
        """

        self.layers = []
        self.weights = []
        self.loss = None
        self.loss_str = ""
        self.learning_rate = 0.0
        self.train_data_x = None
        self.train_data_y = None
        self.validation_x = None
        self.validation_y = None
        self.has_validation_data = False
        self.test_data_x = None
        self.test_data_y = None
        self.epochs = None
        self.callbacks = []

        self.error_val = 0.0
        self.accuracy_val = 0.0

        self.metrics = {'loss' : 0.0, 'acc' : 0.0, 'val_loss' : 0.0, 'val_acc' : 0.0}
        self.current_epoch = 0
        self.is_fitting = True

        self.parameters = 0

        self.is_compiled = False

        if "layers" in kwargs :
            layers_arg = kwargs.get("layers")
            for layer in layers_arg :
                self.add(layer)


    def add(self, layer : BaseLayer) -> None :
        """Method to add layers to the model

        Args
        ----
            layer : BaseLayer
                layer to add to the model
        
        """
        self.layers.append(layer)

    def compile(self) -> None :
        """Model compilation method
        
        To be implemented in child
        """
        pass

    def fit(self) -> None :
        """Method to train the model on data
        
        To be implemented in child
        """
        pass

    def stop(self) -> None :
        """Method to stop the training"""
        self.is_fitting = False

    def summary(self) -> None :
        """Prints a summary of the model with all necessary details"""
        pass

    def save(self) -> None :
        """Saves the model to the disk"""
        pass
    
    def evaluate(self) -> None :
        """Evaluates metrics based on a Test dataset"""
        pass
    
    def predict(self) -> None :
        """Calculate inference using the trained model"""
        pass
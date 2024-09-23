"""Several common layers used in Neural Networks"""

import numpy as np
import cupy as cp

from shamboflow import IS_CUDA
from shamboflow.engine.base_layers import BaseLayer
from shamboflow.engine.activations import get

class Dense(BaseLayer) :
    """A Simple 1D layer

    A Dense layer is a simple 1D layer
    that just has a given number of
    perceptrons. Its the most common
    and basic layer.

    Attributes
    ----------
        size : int
            The number of perceptrons in the layer
        bias : ndarray
            An array of bias values for a perceptron
        activation : function
            The activation function to apply to this layer
        output : ndarray
            An array of output values after applying activation function
    
    """

    def __init__(self, size : int, activation : str, **kwargs) -> None:
        """Constructor for Dense Layer

        Args
        ----
            size : int
                The number of perceptrons in the layer
            activation : str
                The activation function to use for the layer
        """
        super().__init__("Dense", True)

        self.size = size
        self.activation = get(activation)

        self.bias_array = None
        self.output_array = None
        self.leakyrelu_slope = None

        if "leakyrelu_slope" in kwargs : 
            self.leakyrelu_slope = kwargs.get("leakyrelu_slope")
        

    def build(self) -> None:
        """Overidden Build method

        This method initializes the bias and output data array.
        """
        if IS_CUDA :
            self.bias_array = cp.random.rand(self.size)
            self.output_array = cp.random.rand(self.size)
        else :
            self.bias_array = np.random.rand(self.size)
            self.output_array = np.random.rand(self.size)

        super().build()
    
    def compute(self, input : np.ndarray) -> np.ndarray :
        """Method to perform computation on data

        This method accepts an input vector
        that is the output vector of the
        previous layer in the network. Then
        output values of this layer is calculated.

        The input values are simply added with the
        bias and then passed through the activation
        function.

        Args
        ----
            input : ndarray
                The input vector
        
        Returns
        -------
            The output vector after computaion
        
        """

        if IS_CUDA :
            input_gpu = cp.asarray(input)
            midway = cp.add(input_gpu, self.bias_array)
            res = self.activation(cp.asnumpy(midway), self.leakyrelu_slope)
            return res

        midway = np.add(input, self.bias_array)
        res = self.activation(midway, self.leakyrelu_slope)
        return res
    
    def backprop(self) :
        ...
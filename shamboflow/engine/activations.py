"""Activation functions"""

import math
import numpy as np

def signmoid(x : np.ndarray) -> np.ndarray :
    """Sigmoid acitvation function
    
    Applies the sigmoid activation function to a value.
    It returns a value between 0 and 1 but never 0 and 1.
    It suffers from the vanishing gradient issue.

    `sigmoid(x) = 1 / (1 + exp(-x))`

    Range: (0, 1)

    Args
    ----
    x : ndarray
        The input vector to apply activation function over.

    Returns
    -------
        Value after sigmoid is applied on x
    
    """

    return (1 / (1 + np.exp(-x)))

def tanh(x : np.ndarray) -> np.ndarray :
    """Hyperbolic tangent activation function

    `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

    Range: (-1, 1)

    Args
    ----
    x : ndarray
        The input vector to apply activation function over.

    Returns
    -------
        Value after tanh is applied on x
    
    """
    
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

def relu(x : np.ndarray) -> np.ndarray :
    """ReLU activation function
    
    Applies the ReLU activation function on the input x.

    ReLU or Rectilinear Activation Unit was devised to overcome
    the issue of vanishing gradient in sigmoid function. It resolves
    the issue by not having any upper limit. But that comes with
    its own problems.

    `ReLU(x) = max(0, x)`

    Range: [0, inf)

    Args
    ----
    x : ndarray
        The input vector to apply activation function over.

    Returns
    -------
        Value after relu is applied on x
    """

    return np.fmax(0, x)

def leakyrelu(x : float, slope : float) -> float :
    """Leaky ReLU activation function
    
    This function was devised to address
    the issue of negative values as input
    to ReLU. ReLU would simply discard
    them, but leaky relu uses them as well.

    `leakyReLY(x, slope) = if (x > 0) => x, if (x <= 0) => slope * x`

    Range: (-inf, inf)

    Args
    ----
    x : float
        The input value to apply activation function over.
    slope : float
        slope of the line that provides output for negative inputs

    Returns
    -------
        Value after sigmoid is applied on x
    """

    return np.where(x > 0, x, np.multiply(slope, x))

def softmax(x : any) :
    ...
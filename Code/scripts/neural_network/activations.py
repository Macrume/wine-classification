from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from utils.types import FloatArray

ActivationFunction = Callable[[FloatArray], FloatArray]

@dataclass
class Activation:
    """Store an activation function and its derivative.

    The class wraps a callable activation function together with its derivative.
    Instances can be called directly, for example ``relu(x)``, because
    ``__call__`` delegates to the stored activation function.

    Attributes:
        name: Activation function name.
        function: Callable that computes the activation value.
        derivative: Callable that computes the derivative of the activation
            function.
    """
    name: str
    function: ActivationFunction
    derivative: ActivationFunction
   
    def __call__(self, x: FloatArray) -> FloatArray:
        """Return the activation value for the input array."""
        return self.function(x)
    
def sigmoid_function(x: FloatArray) -> FloatArray:
    """Return the sigmoid activation value."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: FloatArray) -> FloatArray:
    """Return the derivative of the sigmoid activation function."""
    sigmoid_value = sigmoid_function(x)
    return sigmoid_value * (1 - sigmoid_value)


def relu_function(x: FloatArray) -> FloatArray:
    """Return the relu activation value."""
    return x * (x > 0)


def relu_derivative(x: FloatArray) -> FloatArray:
    """Return the derivative of the relu activation function."""
    return (x > 0).astype(np.float64)


def linear_function(x: FloatArray) -> FloatArray:
    """Return the linear activation value."""
    return x


def linear_derivative(x: FloatArray) -> FloatArray:
    """Return the derivative of the linear activation function."""
    return np.ones_like(x)

    
relu = Activation(
    name="relu",
    function=relu_function,
    derivative=relu_derivative,
)

linear = Activation(
    name="linear",
    function=linear_function,
    derivative=linear_derivative,
)

sigmoid = Activation(
    name="sigmoid",
    function=sigmoid_function,
    derivative=sigmoid_derivative,
)
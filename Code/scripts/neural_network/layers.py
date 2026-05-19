from __future__ import annotations

import numpy as np

from utils.types import FloatArray
from neural_network.activations import Activation, linear

class DenseLayer:
    """Represent a fully connected neural network layer.

    The layer computes a linear transformation followed by an activation
    function:
    
    ``output = activation(x @ weights + biases)``

    During the forward pass, intermediate values are stored and later reused
    during backpropagation.

    Args:
        num_outputs: Number of output neurons.
        num_inputs: Number of input features.
        activation: Activation function applied after the linear transformation.
        rng: Random number generator used to initialize weights. If None, a new generator is created.

    Attributes:
        num_inputs: Number of input features.
        num_outputs: Number of output neurons.
        activation: Activation function used by the layer.
        weights: Weight matrix with shape ``(num_inputs, num_outputs)``.
        biases: Bias vector with shape ``(num_outputs)``.
        last_input: Input from the most recent forward pass.
        last_z: Linear output from the most recent forward pass.
        last_output: Activated output from the most recent forward pass.
        d_weights: Gradient of the loss with respect to ``weights``.
        d_biases: Gradient of the loss with respect to ``biases``.
    """
    def __init__(
        self,
        num_outputs: int, 
        num_inputs : int,
        activation : Activation = linear,
        rng: np.random.Generator | None = None,
    ) -> None:
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation
        
        if rng is None:
            rng = np.random.default_rng()
        
        self.weights: FloatArray = rng.standard_normal(
            size=(num_inputs, num_outputs)
        ) * 0.01
        self.biases: FloatArray = np.zeros(num_outputs)
        
        self.last_input: FloatArray | None = None
        self.last_z: FloatArray | None = None
        self.last_output: FloatArray | None = None
        
        self.d_weights: FloatArray | None = None
        self.d_biases: FloatArray | None = None
        
        
    def forward(self, x: FloatArray) -> FloatArray:
        """Compute the layer output for a batch of inputs.

        Args:
            x: Input matrix with shape ``(batch_size, num_inputs)``.

        Returns:
            Output matrix with shape ``(batch_size, num_outputs)``.

        Raises:
            ValueError: If ``x`` is not 2D or if its last dimension does not
                match ``num_inputs``.
        """
        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D input with shape (batch_size, {self.num_inputs}), "
                f"got shape {x.shape}"
            )
        
        if x.shape[-1] != self.num_inputs:
            raise ValueError(
                f"Expected input dimension {self.num_inputs}, got {x.shape[-1]}"
            )

        self.last_input = x

        z = x @ self.weights + self.biases
        self.last_z = z

        output = self.activation(z)
        self.last_output = output

        return output
    
    def backward(self, d_output: FloatArray) -> FloatArray:
        """Backpropagate the gradient through the layer.

        The method computes gradients with respect to weights and biases and
        returns the gradient with respect to the layer input.

        Args:
            d_output: Gradient of the loss with respect to the layer output,
                with shape ``(batch_size, num_outputs)``.

        Returns:
            Gradient of the loss with respect to the layer input, with shape
            ``(batch_size, num_inputs)``.

        Raises:
            RuntimeError: If called before ``forward``.
        """
        if self.last_input is None or self.last_z is None:
            raise RuntimeError("Cannot call backward before forward")
        
        d_z = d_output * self.activation.derivative(self.last_z)
        
        batch_size = self.last_input.shape[0]

        self.d_weights = self.last_input.T @ d_z 
        self.d_biases = np.sum(d_z, axis=0) 

        d_input = d_z @ self.weights.T

        return d_input


    def __call__(self, x: FloatArray) -> FloatArray:
        """Compute the layer output for a batch of inputs."""
        return self.forward(x)
        
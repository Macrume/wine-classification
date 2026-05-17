from __future__ import annotations

import numpy as np

from utils.types import FloatArray
from neural_network.activations import Activation, linear

class DenseLayer():
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
        
        
    def forward(self, x: FloatArray) -> FloatArray:
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
        if self.last_input is None or self.last_z is None:
            raise RuntimeError("Cannot call backward before forward")
        
        d_z = d_output * self.activation.derivative(self.last_z)
        
        batch_size = self.last_input.shape[0]

        self.d_weights = self.last_input.T @ d_z 
        self.d_biases = np.sum(d_z, axis=0) 

        d_input = d_z @ self.weights.T

        return d_input


    def __call__(self, x: FloatArray) -> FloatArray:
         return self.forward(x)
        

from __future__ import annotations

from neural_network.mlp import MultiLayerPerceptron


class SGD:
    """Stochastic gradient descent optimizer.

    Updates model parameters using gradients computed during backpropagation.

    Attributes:
        learning_rate: Step size used during parameter updates.
    """
    def __init__(self, learning_rate: float = 0.01) -> None:
        """Initialize the SGD optimizer.

        Args:
            learning_rate: Step size used to update model parameters.

        Raises:
            ValueError: If ``learning_rate`` is not positive.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = learning_rate

    def step(self, model: MultiLayerPerceptron) -> None:
        """Update all trainable parameters of the model.

        Args:
            model: Model whose layer parameters should be updated.

        Raises:
            RuntimeError: If gradients are missing. This usually means that
                ``backward`` was not called before ``step``.
        """
        for layer in model.layers:
            if layer.d_weights is None or layer.d_biases is None:
                raise RuntimeError("Cannot update before backward")

            layer.weights -= self.learning_rate * layer.d_weights
            layer.biases -= self.learning_rate * layer.d_biases
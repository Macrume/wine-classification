
from __future__ import annotations

from neural_network.mlp import MultiLayerPerceptron


class SGD:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step(self, model: MultiLayerPerceptron) -> None:
        for layer in model.layers:
            if layer.d_weights is None or layer.d_biases is None:
                raise RuntimeError("Cannot update before backward")

            layer.weights -= self.learning_rate * layer.d_weights
            layer.biases -= self.learning_rate * layer.d_biases
from neural_network.types import FloatArray
from neural_network.layers import DenseLayer
from neural_network.activations import Activation, relu, linear


class MultiLayerPerceptron:
    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[Activation] | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Provide at least input and output layer size")

        num_layers = len(layer_sizes) - 1

        if activations is None:
            activations = [relu] * (num_layers - 1) + [linear]

        if len(activations) != num_layers:
            raise ValueError(
                f"Expected {num_layers} activations, got {len(activations)}"
            )

        self.layer_sizes = layer_sizes
        self.layers: list[DenseLayer] = []

        for i in range(num_layers):
            layer = DenseLayer(
                num_inputs=layer_sizes[i],
                num_outputs=layer_sizes[i + 1],
                activation=activations[i],
            )
            self.layers.append(layer)

    def forward(self, x: FloatArray) -> FloatArray:
        for layer in self.layers:
            x = layer(x)

        return x

    def __call__(self, x: FloatArray) -> FloatArray:
        return self.forward(x)
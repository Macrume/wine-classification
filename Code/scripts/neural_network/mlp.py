import numpy as np

from utils.types import FloatArray
from neural_network.layers import DenseLayer
from neural_network.activations import Activation, relu, linear


class MultiLayerPerceptron:
    """Multilayer perceptron model composed of dense layers.

    The model consists of a sequence of fully connected layers. Each layer
    applies a linear transformation followed by an activation function.

    Attributes:
        layer_sizes: List of layer sizes. For example, ``[13, 16, 8, 3]``
            creates three dense layers: ``13 -> 16``, ``16 -> 8`` and
            ``8 -> 3``.
        layers: List of dense layers used by the model.
    """
    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[Activation] | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialize a multilayer perceptron.

        Args:
            layer_sizes: Sizes of consecutive layers. The first value is the
                input dimension and the last value is the output dimension.
            activations: Activation functions for each dense layer. If ``None``,
                ReLU is used for hidden layers and linear activation is used
                for the output layer.
            random_state: Seed used for deterministic weight initialization.

        Raises:
            ValueError: If fewer than two layer sizes are provided or 
            if the number of activation functions does not match
            the number of dense layers.
        """
        if len(layer_sizes) < 2:
            raise ValueError("Provide at least input and output layer size")

        num_layers = len(layer_sizes) - 1

        if activations is None:
            activations = [relu] * (num_layers - 1) + [linear]

        if len(activations) != num_layers:
            raise ValueError(
                f"Expected {num_layers} activations, got {len(activations)}"
            )
            
        rng = np.random.default_rng(random_state)

        self.layer_sizes = layer_sizes
        self.layers: list[DenseLayer] = []

        for i in range(num_layers):
            layer = DenseLayer(
                num_inputs=layer_sizes[i],
                num_outputs=layer_sizes[i + 1],
                activation=activations[i],
                rng=rng,
            )
            self.layers.append(layer)

    def forward(self, x: FloatArray) -> FloatArray:
        """Compute model output for a batch of input samples.

        Args:
            x: Input data with shape ``(batch_size, num_features)``.

        Returns:
            Raw model outputs with shape
            ``(batch_size, output_dimension)``.
        """
        for layer in self.layers:
            x = layer(x)

        return x
    
    def backward(self, d_output: FloatArray ) -> FloatArray:
        """Run backpropagation through all layers.

        Args:
            d_output: Gradient of the loss with respect to the model output.
                Shape should match the output of the last layer.

        Returns:
            Gradient of the loss with respect to the model input.
        """
        gradient = d_output
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    

    def __call__(self, x: FloatArray) -> FloatArray:
        """Compute model output for a batch of input samples.

        Args:
            x: Input data with shape ``(batch_size, num_features)``.

        Returns:
            Raw model outputs with shape
            ``(batch_size, output_dimension)``.
        """
        return self.forward(x)
import numpy as np

from neural_network import MultiLayerPerceptron, relu, linear


x = np.random.rand(12)

mlp = MultiLayerPerceptron(
    layer_sizes=[12, 8, 3],
    activations=[relu, linear],
)

y = mlp(x)

print(y)
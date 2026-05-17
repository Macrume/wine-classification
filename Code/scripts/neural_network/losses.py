import numpy as np

from utils.types import FloatArray, IntArray

import numpy as np

def softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class MeanSquaredError:
    def loss(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def derivative(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        return 2 * (y_pred - y_true) / y_true.size
    
class CrossEntropyLoss:
    def loss(self, logits: FloatArray, y_true: IntArray) -> float:
        probabilities = softmax(logits)

        batch_size = logits.shape[0]

        correct_class_probabilities = probabilities[
            np.arange(batch_size),
            y_true,
        ]

        epsilon = 1e-15
        correct_class_probabilities = np.clip(
            correct_class_probabilities,
            epsilon,
            1.0,
        )

        return float(-np.mean(np.log(correct_class_probabilities)))

    def derivative(self, logits: FloatArray, y_true: IntArray) -> FloatArray:
        probabilities = softmax(logits)

        batch_size = logits.shape[0]

        probabilities[np.arange(batch_size), y_true] -= 1
        probabilities /= batch_size

        return probabilities
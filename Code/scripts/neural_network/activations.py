from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.types import FloatArray

ActivationFunction = Callable[[FloatArray], FloatArray]

@dataclass
class Activation:
    name: str
    function: ActivationFunction
    derivative: ActivationFunction
   
    def __call__(self, x: FloatArray) -> FloatArray:
        return self.function(x)
    
relu = Activation(
    name="relu",
    function=lambda x: x * (x > 0),
    derivative=lambda x: (x > 0).astype(np.float64),
)

linear = Activation(
    name="linear",
    function=lambda x: x,
    derivative=lambda x: np.ones_like(x),
)

sigmoid = Activation(
    name="sigmoid",
    function=lambda x: 1 / (1 + np.exp(-x)),
    derivative=lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x)))),
)
"""
File containing the dense layer class
"""

import numpy as np
from net.layer import Layer
from net.tensor import Tensor

class Dense(Layer):
    """
    A class representing a dense layer in a neural network,
    inheriting from the Layer class
    """

    weights: Tensor
    biases: Tensor

    def __init__(self, inp: int, out: int):
        """
        Initialize a dense layer which of given dimensions
        """
        super().__init__()
        self.weights = Tensor(np.random.randn(out, inp) * np.sqrt(2 / inp))
        self.biases = Tensor(np.zeros((1, out))) # For broadcasting

    def forward(self, inp: Tensor) -> None:
        """
        Forward pass for a dense layer
        """
        if len(inp.shape) == 1:
            inp.reshape((1, -1)) # May not need this??

        self.inp = inp
        self.out = Tensor(inp.value @ self.weights.value.T + self.biases.value)

    def backward(self) -> None:
        """
        Backward pass for a dense layer, assuming out has already been backpropagated through
        """
        self.weights.grad = self.out.grad.T @ self.inp.value # Implicit sum across batch
        self.biases.grad = np.sum(self.out.grad, axis=0, keepdims=True) # Explicit sum
        self.inp.grad = self.out.grad @ self.weights.value

    def parameters(self):
        """
        Return the weights and biases of the layer
        """
        return [self.weights, self.biases]

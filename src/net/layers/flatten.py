"""
File containing the flatten layer class
"""

import numpy as np
from net.layer import Layer
from net.tensor import Tensor

class Flatten(Layer):
    """
    A flattening layer for moving from a convolutional to a dense layer
    """

    def forward(self, inp: Tensor) -> None:
        """
        Forward for the flatten layer
        """
        self.inp = inp

        if len(inp.shape) == 1:
            self.out = inp.flatten()
        else:
            self.out = inp.reshape_(shape=(inp.shape[0], -1))

    def backward(self):
        """
        Backward for flatten layer
        """
        self.inp.grad = np.reshape(self.out.grad, self.inp.shape)

    def parameters(self):
        """
        Layer has no parameters
        """
        return []

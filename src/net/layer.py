"""
The file containing the Layer class
"""

import numpy as np
from net.tensor import Tensor

class Layer:
    """
    A class representing an abstract layer in a neural network
    """

    inp: Tensor
    out: Tensor

    def __init__(self, inp: np.ndarray):
        self.inp = Tensor(inp)
        self.out = Tensor(None)

    def forward(self):
        """
        A method representing the forward pass through a layer
        """
        raise NotImplementedError

    def backward(self):
        """
        A method representing the backward pass through a layer
        """
        raise NotImplementedError

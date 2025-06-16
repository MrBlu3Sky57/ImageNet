"""
File containing the network class
"""

import numpy as np
from net.layer import Layer
from net.tensor import Tensor

class Network:
    """
    Class representing a neural network
    """
    layers = list[Layer]

    def __init__(self, layers):
        """ Requires layers to be inputted
        in sequential order, with correct dimensions
        """
        self.layers = layers

    def forward(self, inp: np.ndarray) -> Tensor:
        """ Forward pass through the network"""

        if len(inp.shape) == 1:
            inp = inp.reshape(1, inp.shape[0])
        tensor_inp = Tensor(inp)
        for layer in self.layers:
            layer.forward(tensor_inp)
            tensor_inp = layer.out
        return tensor_inp

    def backward(self) -> None:
        """Backward pass through the network, 
        forward must have been called first"""

        for layer in reversed(self.layers):
            layer.backward()

    def parameters(self) -> list[Tensor]:
        """
        Get the parameters of the network
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

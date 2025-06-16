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
    layers: list[Layer]
    training: bool

    def __init__(self, layers: list[Layer], training: bool = True):
        """ Requires layers to be inputted
        in sequential order, with correct dimensions
        """
        self.layers = layers
        self.training = training
        for layer in layers:
            layer.training = training

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
    
    def set_to_predict(self) -> None:
        """
        Set model to prediction mode
        """
        self.training = False
        for layer in self.layers:
            layer.training = False

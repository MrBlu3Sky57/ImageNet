"""
The file containing the Layer class
"""

from net.tensor import Tensor

class Layer:
    """
    A class representing an abstract layer in a neural network
    """

    inp: Tensor
    out: Tensor
    training: bool

    def __init__(self, training: bool = True):
        self.inp = Tensor(None)
        self.out = Tensor(None)
        self.training = training

    def forward(self, inp: Tensor) -> None:
        """
        A method representing the forward pass through a layer
        """
        raise NotImplementedError

    def backward(self) -> None:
        """
        A method representing the backward pass through a layer
        """
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        """
        A method that returns the layer's parameters as a list of Tensors
        """
        raise NotImplementedError

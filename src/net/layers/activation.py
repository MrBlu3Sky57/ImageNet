"""
File containing the activation layer class
"""

from net.layer import Layer
from net.tensor import Tensor

class Activation(Layer):
    """
    A class representing an activation layer in a neural network,
    inheriting from the Layer class
    """

    act: callable
    d_act: callable

    def __init__(self, act: callable, d_act: callable):
        """
        Initialize an activation layer with the given dimensions
        """
        super().__init__()
        self.act = act
        self.d_act = d_act

    def forward(self, inp: Tensor) -> None:
        """
        Forward pass for a dense layer
        """
        if len(inp.shape) == 1:
            inp.reshape((1, -1)) # May not need this??
        self.inp = inp
        self.out = self.act(inp)

    def backward(self) -> None:
        """
        Backward pass for a dense layer, assuming out has already been backpropagated through
        """
        self.inp.grad = self.d_act(self.inp) * self.out.grad

    def parameters(self):
        """
        This layer has no parameters
        """
        return []

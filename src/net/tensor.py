"""
File containing the Tensor Class
"""


import numpy as np

class Tensor():
    """
    Tensor class that stores value and grad arrays meant to represent
    a matrix or vector in a neural network
    """
    value: np.ndarray
    grad: np.ndarray
    shape: tuple[int]
    shape_hint: tuple[int]

    def __init__(self, value: np.ndarray, shape_hint: tuple = None):
        if value is not None:
            self.shape_hint = shape_hint or value.shape
            self.shape = value.shape
            self.value = value
            self.grad = np.zeros_like(value)
        else:
            self.shape_hint = None
            self.shape = None
            self.value = None
            self.grad = None

    def zero_grad(self):
        """
        Set the gradients to zero
        """
        self.grad = np.zeros(self.value.shape)

    def increment(self, lr):
        """
        Increment the value based on the gradient with the given learning
        rate
        """
        self.value -= lr * self.grad

    def reshape(self, shape: tuple[int]) -> None:
        """
        Reshape tensor to given shape
        """

        self.value = self.value.reshape(shape)
        self.grad = self.grad.reshape(shape)
        self.shape = shape

    def reshape_(self, shape: tuple[int]):
        """
        Reshape tensor to given shape and return a new one
        """
        new = Tensor(self.value.reshape(shape))
        new.grad = self.grad.reshape(shape)
        return new

    def flatten(self):
        """
        Flatten tensor
        """
        new = Tensor(self.value.flatten())
        new.grad = self.grad.flatten()
        return new

    def restore_shape(self) -> None:
        """
        Restore shape to shape hint
        """
        self.reshape(self.shape_hint)
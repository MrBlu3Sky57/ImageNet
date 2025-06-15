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

    def __init__(self, value: np.ndarray):
        if value is not None:
            self.shape = value.shape
            self.value = value
            self.grad = np.zeros(value.shape)
        else:
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

    def flatten(self):
        """
        Flatten tensor
        """
        new = Tensor(self.value.flatten())
        new.grad = self.grad.flatten()
        return new
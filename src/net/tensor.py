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

    def __init__(self, value: np.ndarray):
        if value is not None:
            self.value = value
            self.grad = np.zeros(value.shape)
        else:
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

"""
Module containing utility functions
"""

import numpy as np
from net.tensor import Tensor

def relu(x: Tensor):
    """
    ReLU activation function
    """
    return Tensor(np.maximum(0, x.value))

def d_relu(x: Tensor):
    """
    ReLU derivative
    """
    return (x.value > 0).astype(float)

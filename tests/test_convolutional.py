"""
Test file for the convolutional layer
--- Written by Chat GPT ---
"""

import numpy as np
from net.tensor import Tensor
from net.layers import Convolutional
from net.layers import Activation
from net.network import Network
from net.util import relu, d_relu

def test_convolutional_forward_shapes():
    # Input: batch of 2 images, 1 channel, 5x5
    x = np.random.randn(2, 1, 5, 5)
    x_tensor = Tensor(x)

    # Kernel: 2 filters, 1 in-channel, 3x3 kernel
    kernel = Tensor(np.random.randn(2, 1, 3, 3), shape_hint=(2, 1, 3, 3))

    # Create Conv Layer
    conv = Convolutional(kernel, strides=1, padding=1)

    # Forward
    conv.forward(x_tensor)

    # Should preserve spatial dimensions due to padding
    assert conv.out.value.shape == (2, 2, 5, 5)

def test_convolutional_backward_shapes():
    x = np.random.randn(2, 1, 5, 5)
    x_tensor = Tensor(x)

    kernel = Tensor(np.random.randn(2, 1, 3, 3), shape_hint=(2, 1, 3, 3))
    conv = Convolutional(kernel, strides=1, padding=1)

    conv.forward(x_tensor)

    # Simulate a loss gradient w.r.t. output
    grad = np.random.randn(*conv.out.value.shape)
    conv.out.grad = grad

    # Backward
    conv.backward()

    # Check shapes
    assert conv.kernel.grad.shape == (2, 9)
    assert conv.inp.grad.shape == (2, 1, 5, 5)

def test_convolutional_in_network():


    kernel = Tensor(np.random.randn(2, 1, 3, 3), shape_hint=(2, 1, 3, 3))
    conv = Convolutional(kernel, strides=1, padding=1)
    act = Activation(relu, d_relu)

    net = Network([conv, act])

    x = np.random.randn(1, 1, 5, 5)
    output = net.forward(x)

    assert output.value.shape == (1, 2, 5, 5)

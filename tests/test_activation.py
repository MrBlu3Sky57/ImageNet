"""
Test the activation layer class
--- Test was written by Chat GPT
"""


import numpy as np
from net.tensor import Tensor
from net.layers.activation import Activation
from net.util import relu, d_relu


def test_activation_forward_relu():
    act = Activation(relu, d_relu)
    x = Tensor(np.array([[-1.0, 0.0, 2.0], [3.0, -2.0, 1.0]]))  # shape (2, 3)
    act.forward(x)
    expected = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 1.0]])
    np.testing.assert_array_equal(act.out.value, expected)

def test_activation_backward_relu():
    act = Activation(relu, d_relu)
    x = Tensor(np.array([[-1.0, 2.0], [3.0, -4.0]]))
    act.forward(x)
    act.out.grad = np.ones_like(act.out.value)  # upstream gradient of 1s
    act.backward()
    expected_grad = np.array([[0.0, 1.0], [1.0, 0.0]])  # d_relu mask
    np.testing.assert_array_equal(act.inp.grad, expected_grad)

def test_activation_parameters_is_none():
    act = Activation(relu, d_relu)
    assert act.parameters() == []
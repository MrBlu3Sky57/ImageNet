"""
Test the dense layer class
--- Test was written by Chat GPT
"""

import numpy as np
from net.tensor import Tensor
from net.layers import Dense

def test_dense_forward_shape():
    layer = Dense(inp=4, out=3)
    x = Tensor(np.random.randn(5, 4))  # batch of 5
    layer.forward(x)
    out = layer.out
    assert out.value.shape == (5, 3)

def test_dense_forward_single_input():
    layer = Dense(3, 2)
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    layer.forward(x)
    out = layer.out
    assert out.value.shape in [(1, 2), (2,)]

def test_dense_backward_shapes():
    layer = Dense(inp=4, out=2)
    x = Tensor(np.random.randn(6, 4))
    layer.forward(x)

    # Simulate gradient flowing from next layer
    layer.out.grad = np.ones_like(layer.out.value)
    layer.backward()

    assert layer.weights.grad.shape == (2, 4)
    assert layer.biases.grad.shape == (2,)
    assert layer.inp.grad.shape == (6, 4)

def test_dense_parameters():
    layer = Dense(inp=3, out=2)
    params = layer.parameters()
    assert len(params) == 2
    assert all(isinstance(p, Tensor) for p in params)
    assert params[0].value.shape == (2, 3)
    assert params[1].value.shape == (2,) or params[1].value.shape == (1, 2)

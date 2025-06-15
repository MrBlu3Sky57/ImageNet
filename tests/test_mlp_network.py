"""
Test the network class on only the dense and activation layer classes
--- Test was written by Chat GPT ---
"""

import numpy as np
from net.network import Network
from net.layers.dense import Dense
from net.layers.activation import Activation
from net.tensor import Tensor
from net.util import relu, d_relu

def build_mlp():
    """Helper to build a simple 2-layer MLP"""
    return Network([
        Dense(4, 5),
        Activation(relu, d_relu),
        Dense(5, 3)
    ])

def test_network_forward_batch():
    model = build_mlp()
    x = np.random.randn(2, 4)  # batch of 2 examples
    out = model.forward(x)
    assert isinstance(out, Tensor)
    assert out.value.shape == (2, 3)

def test_network_forward_single_example():
    model = build_mlp()
    x = np.random.randn(4)  # single example
    out = model.forward(x)
    assert out.value.shape in [(1, 3), (3,)]

def test_network_backward_and_grads():
    model = build_mlp()
    x = np.random.randn(3, 4)
    out = model.forward(x)
    out.grad = np.ones_like(out.value)  # simulate upstream gradient
    model.backward()

    # Check input gradient is populated
    inp_grad = model.layers[0].inp.grad
    assert inp_grad.shape == (3, 4)

    # Check parameter gradients
    for param in model.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.value.shape

def test_network_parameters_returns_all():
    model = build_mlp()
    params = model.parameters()
    assert len(params) == 4  # weights and biases for 2 Dense layers
    for p in params:
        assert isinstance(p, Tensor)

def mse_loss(pred: Tensor, target: np.ndarray) -> float:
    """Mean squared error loss"""
    return 0.5 * np.sum((pred.value - target) ** 2)

def mse_grad(pred: Tensor, target: np.ndarray) -> np.ndarray:
    """Gradient of MSE wrt prediction"""
    return pred.value - target

def test_gradient_correctness_numerical():
    np.random.seed(42)
    eps = 1e-5
    model = Network([
        Dense(4, 5),
        Activation(relu, d_relu),
        Dense(5, 1)
    ])

    # Single input and target
    x = np.random.randn(4)
    y_true = np.random.randn(1)

    # Forward pass
    out = model.forward(x)
    loss = mse_loss(out, y_true)

    # Backward pass
    out.grad = mse_grad(out, y_true)
    model.backward()

    # Numerical gradient check
    for param in model.parameters():
        numerical_grad = np.zeros_like(param.grad)
        for idx in np.ndindex(*param.value.shape):
            original = param.value[idx]

            param.value[idx] = original + eps
            loss_plus = mse_loss(model.forward(x), y_true)

            param.value[idx] = original - eps
            loss_minus = mse_loss(model.forward(x), y_true)

            param.value[idx] = original  # restore

            numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        # Compare gradients
        np.testing.assert_allclose(param.grad, numerical_grad, rtol=1e-2, atol=1e-4)
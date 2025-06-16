"""
Test file for backpropagation
--- Written by Chat GPT ---
"""

import numpy as np
from net.layers.convolution import Convolutional
from net.layers.flatten import Flatten
from net.network import Network
from net.layers import Dense, Activation
from net.tensor import Tensor
from net.util import relu, d_relu, cross_entropy

def test_network_gradient_check():
    np.random.seed(0)

    # Simple MLP: Dense -> ReLU -> Dense
    net = Network([
        Dense(2, 3),
        Activation(relu, d_relu),
        Dense(3, 2)
    ])

    # Small input and target
    x = np.array([[0.5, -0.3]])
    y = np.array([1])  # Class index (e.g., for softmax cross-entropy)

    # Forward pass
    out = net.forward(x)
    loss, grad = cross_entropy(out.value, y)
    out.grad = grad
    net.backward()

    # Check gradients using finite differences
    epsilon = 1e-5
    for p in net.parameters():
        analytical_grad = p.grad
        numerical_grad = np.zeros_like(p.value)

        for i in np.ndindex(*p.value.shape):
            orig = p.value[i]

            p.value[i] = orig + epsilon
            out_plus = net.forward(x)
            loss_plus, _ = cross_entropy(out_plus.value, y)

            p.value[i] = orig - epsilon
            out_minus = net.forward(x)
            loss_minus, _ = cross_entropy(out_minus.value, y)

            numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            p.value[i] = orig  # Restore

        # Assert gradients are close
        assert np.allclose(
            analytical_grad, numerical_grad, atol=1e-4
        ), f"Gradient check failed for param with shape {p.value.shape}"

def test_cnn_gradient_check():
    np.random.seed(0)

    # Simple CNN: Conv -> ReLU -> Flatten -> Dense
    kernel = Tensor(np.random.randn(2, 1, 3, 3))  # 2 filters, 1 input channel, 3x3 kernels
    net = Network([
        Convolutional(kernel, strides=1, padding=1),
        Activation(relu, d_relu),
        Flatten(),
        Dense(2 * 5 * 5, 3)  # Assuming input shape is (1, 1, 5, 5)
    ])

    # Small input and target
    x = np.random.randn(1, 1, 5, 5)  # batch_size=1, 1 channel, 5x5 image
    y = np.array([2])  # Class index for softmax cross-entropy

    # Forward pass
    out = net.forward(x)
    loss, grad = cross_entropy(out.value, y)
    out.grad = grad
    net.backward()

    # Check gradients using finite differences
    epsilon = 1e-5
    for p in net.parameters():
        analytical_grad = p.grad
        numerical_grad = np.zeros_like(p.value)

        for i in np.ndindex(*p.value.shape):
            orig = p.value[i]

            p.value[i] = orig + epsilon
            out_plus = net.forward(x)
            loss_plus, _ = cross_entropy(out_plus.value, y)

            p.value[i] = orig - epsilon
            out_minus = net.forward(x)
            loss_minus, _ = cross_entropy(out_minus.value, y)

            numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            p.value[i] = orig  # Restore

        # Assert gradients are close
        assert np.allclose(
            analytical_grad, numerical_grad, atol=1e-4
        ), f"Gradient check failed for param with shape {p.value.shape}"
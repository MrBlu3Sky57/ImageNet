"""
Test file for backpropagation
--- Written by Chat GPT ---
"""

import numpy as np
from net.layers.convolution import Convolutional
from net.layers.flatten import Flatten
from net.network import Network
from net.layers import Dense, Activation, BatchNorm
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

def numerical_gradient(f, x, eps=1e-8):
    """Compute numerical gradient of f at x (shape: (n, c, h, w))"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        x[idx] += eps
        fx1 = f(x)
        x[idx] -= 2 * eps
        fx2 = f(x)
        x[idx] += eps
        grad[idx] = (fx1 - fx2) / (2 * eps)
        it.iternext()
    return grad

def test_batchnorm_backward():
    np.random.seed(42)
    n, c, h, w = 4, 3, 5, 5
    x = np.random.randn(n, c, h, w).astype(np.float64)

    bn = BatchNorm(channels=c)
    bn.gamma = Tensor(np.random.randn(bn.gamma.shape[0], bn.gamma.shape[1]))
    bn.beta = Tensor(np.random.randn(bn.beta.shape[0], bn.beta.shape[1]))
    bn.training = True

    inp = Tensor(x.copy())
    bn.forward(inp)

    # Set dummy gradient on output
    bn.out.grad = np.random.randn(*bn.out.value.shape)

    # Backward pass
    bn.backward()

    def f_x(x_val):
        bn_temp = BatchNorm(channels=c)
        bn_temp.training = True

        # Copy gamma and beta from original layer
        bn_temp.gamma.value = bn.gamma.value.copy()
        bn_temp.beta.value = bn.beta.value.copy()

        inp_temp = Tensor(x_val.copy())
        bn_temp.forward(inp_temp)

        # Use the same dummy out.grad for consistency
        bn_temp.out.grad = bn.out.grad.copy()

        return np.sum(bn_temp.out.value * bn_temp.out.grad)


    num_grad = numerical_gradient(f_x, x.copy())
    ana_grad = inp.grad

    # Assert that analytical and numerical gradients are close
    np.testing.assert_allclose(ana_grad, num_grad, rtol=1e-3, atol=1e-5)

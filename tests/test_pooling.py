"""
Test file for pooling layer functionality
--- Written by Chat GPT ---
"""

import numpy as np
from net.tensor import Tensor
from net.network import Network
from net.train import grad_descent
from net.layers.dense import Dense
from net.layers.activation import Activation
from net.layers.convolution import Convolutional
from net.layers.flatten import Flatten
from net.layers.batch_norm import BatchNorm
from net.layers.pool import Pool
from net.util import cross_entropy, relu, d_relu

# Just for testing
def leaky_relu(x, alpha=0.01): return Tensor(np.where(x.value > 0, x.value, alpha * x.value))
def d_leaky_relu(x, alpha=0.01): return np.where(x.value > 0, 1.0, alpha)

def test_maxpool_forward_simple():
    X = np.array([[[[1, 2],
                    [3, 4]]]])  # Shape (1, 1, 2, 2)

    pool = Pool(size=2, stride=2)
    pool.forward(Tensor(X))

    # Only one region, max is 4
    expected = np.array([[[[4]]]])
    assert np.array_equal(pool.out.value, expected), f"Expected {expected}, got {pool.out.value}"

def test_maxpool_backward_simple():
    X = np.array([[[[1, 2],
                    [3, 4]]]], dtype=float)

    t = Tensor(X)
    pool = Pool(size=2, stride=2)
    pool.forward(t)
    pool.out.grad = np.ones_like(pool.out.value)
    pool.backward()

    expected_grad = np.array([[[[0, 0],
                                [0, 1]]]])  # Gradient should flow only to max (4)
    assert np.array_equal(t.grad, expected_grad), f"Expected {expected_grad}, got {t.grad}"

def test_cnn_with_pool_learns_and():
    X = np.array([
        [[[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]],  # 0 OR 0 → 0
        [[[1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]],  # 1 OR 0 → 1
        [[[0,1,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]],  # 0 OR 1 → 1
        [[[1,1,0,0], [1,1,0,0], [0,0,0,0], [0,0,0,0]]]   # 1 OR 1 → 1
    ]).astype(float)
    Y = np.array([0, 1, 1, 1])

    model = Network([
        Convolutional(Tensor(np.random.randn(2, 1, 2, 2) * 0.1), strides=1, padding=0),
        BatchNorm(channels=2),
        Activation(leaky_relu, d_leaky_relu),
        Pool(size=2, stride=2),
        Flatten(),
        Dense(2, 2)
    ])

    grad_descent(model, cross_entropy, X, Y, steps=1000, batches=4, lr=0.1)
    model.set_to_predict()
    preds = np.argmax(model.forward(X).value, axis=1)
    assert np.array_equal(preds, Y), f"Predictions {preds} don't match labels {Y}"
"""
Test gradient descent
--- Test was written by Chat GPT ---
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
from net.util import cross_entropy, relu, d_relu, tanh, dtanh


def test_mlp_learns_xor():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([0, 1, 1, 0])

    model = Network([
        Dense(2, 8),
        Activation(tanh, dtanh),
        Dense(8, 2)
    ])

    grad_descent(model, cross_entropy, X, Y, steps=500, batches=4, lr=0.1)

    logits = model.forward(X).value
    preds = np.argmax(logits, axis=1)
    assert np.array_equal(preds, Y), f"Predictions {preds} don't match labels {Y}"

def test_cnn_learns_or():
    X = np.array([
        [[[0,0],[0,0]]],
        [[[1,0],[0,0]]],
        [[[0,1],[0,0]]],
        [[[1,1],[1,1]]]
    ]).astype(float)  # Shape (4, 1, 2, 2)

    Y = np.array([0, 1, 1, 1])

    kernel = Tensor(np.random.randn(2, 1, 2, 2) * 0.1)  # small init to help ReLU not die
    model = Network([
        Convolutional(kernel, strides=1, padding=0),
        BatchNorm(channels=2),
        Activation(relu, d_relu),
        Flatten(),
        Dense(2, 2)
    ])

    grad_descent(model, cross_entropy, X, Y, steps=1000, batches=4, lr=0.05)

    logits = model.forward(X).value
    for layer in model.layers:
        print(layer.__class__.__name__)
        print("Input shape:", layer.inp.shape)
        print("Output shape:", layer.out.shape)

    preds = np.argmax(logits, axis=1)
    assert np.array_equal(preds, Y), f"Predictions {preds} don't match labels {Y}"


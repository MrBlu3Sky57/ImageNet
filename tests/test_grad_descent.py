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
from net.util import cross_entropy, relu, d_relu, tanh, dtanh, SEED

# Just for testing
def leaky_relu(x, alpha=0.01): return Tensor(np.where(x.value > 0, x.value, alpha * x.value))
def d_leaky_relu(x, alpha=0.01): return np.where(x.value > 0, 1.0, alpha)

def test_mlp_learns_xor():
    np.random.seed(42)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([0, 1, 1, 0])

    model = Network([
        Dense(2, 8),
        Activation(tanh, dtanh),
        Dense(8, 2)
    ])

    grad_descent(model, cross_entropy, X, Y, steps=1000, batches=4, lr=0.03)

    logits = model.forward(X).value
    preds = np.argmax(logits, axis=1)
    assert np.array_equal(preds, Y), f"Predictions {preds} don't match labels {Y}"

def test_dense_overfit_tiny_data():
    np.random.seed(0)
    X = np.random.randn(10, 64)
    Y = np.random.randint(0, 2, size=(10,))

    model = Network([
        Dense(64, 2)
    ])

    grad_descent(model, cross_entropy, X.reshape(10, 64), Y, steps=3000, batches=5, lr=0.1)

    model.set_to_predict()
    logits = model.forward(X.reshape(10, 64)).value
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == Y)
    print(f"Dense-only overfit acc: {acc:.2f}")
    assert acc == 1.0


def test_cnn_learns_xor():
    X = np.array([
        [[[0,0],[0,0]]],
        [[[1,0],[0,0]]],
        [[[0,1],[0,0]]],
        [[[1,1],[1,1]]]
    ]).astype(float)  # Shape (4, 1, 2, 2)

    Y = np.array([0, 1, 1, 0])

    kernel = Tensor(np.random.randn(2, 1, 2, 2) * 0.1)  # small init to help ReLU not die
    model = Network([
        Convolutional(kernel, strides=1, padding=0),
        BatchNorm(channels=2),
        Activation(leaky_relu, d_leaky_relu),
        Flatten(),
        Dense(2, 2)
    ])

    grad_descent(model, cross_entropy, X, Y, steps=1000, batches=4, lr=0.05)
    logits = model.forward(X).value

    preds = np.argmax(logits, axis=1)
    assert np.array_equal(preds, Y), f"Predictions {preds} don't match labels {Y}"

def test_deeper_cnn_learns_synthetic_binary_with_batchnorm():
    np.random.seed(42)

    # Structured synthetic data with stronger signal
    X = np.random.randn(100, 1, 28, 28) * 0.1
    thresholds = np.sum(X, axis=(1, 2, 3))
    Y = (thresholds > 0).astype(int)
    X[Y == 1] += 1.0  # strengthen signal

    # Model
    model = Network([
        Convolutional(Tensor(np.random.randn(8, 1, 3, 3) * 0.1), strides=1, padding=1),
        BatchNorm(channels=8),
        Activation(relu, d_relu),
        Flatten(),
        Dense(8 * 28 * 28, 2)
    ])

    # Train
    grad_descent(model, cross_entropy, X, Y, steps=2000, batches=10, lr=0.05)

    # Predict
    model.set_to_predict()
    logits = model.forward(X).value
    preds = np.argmax(logits, axis=1)
    accuracy = np.mean(preds == Y)

    print(f"Accuracy: {accuracy:.2%}")
    assert accuracy >= 0.9, f"Accuracy only {accuracy:.2%}"


def test_cnn_overfits_tiny_dataset():
    np.random.seed(0)

    # Tiny synthetic dataset (easy to memorize)
    X = np.random.randn(10, 1, 8, 8)  # Smaller than 28x28 to speed up
    Y = np.random.randint(0, 2, size=(10,))

    # CNN model
    model = Network([
        Convolutional(Tensor(np.random.randn(4, 1, 3, 3) * 0.1), strides=1, padding=1),
        BatchNorm(channels=4),
        Activation(relu, d_relu),
        Flatten(),
        Dense(4 * 8 * 8, 2)
    ])

    # Train on tiny dataset
    grad_descent(model, cross_entropy, X, Y, steps=5000, batches=5, lr=0.05)

    # Evaluate
    model.set_to_predict()
    logits = model.forward(X).value
    preds = np.argmax(logits, axis=1)
    accuracy = np.mean(preds == Y)

    print(f"Accuracy on tiny dataset: {accuracy * 100:.2f}%")
    assert accuracy == 1.0, f"Model failed to overfit tiny dataset, got {accuracy:.2f}"
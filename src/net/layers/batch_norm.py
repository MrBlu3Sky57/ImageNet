"""
File containing the batch norm class
"""

import numpy as np
from net.layer import Layer
from net.tensor import Tensor

class BatchNorm(Layer):
    """
    Class representing a batch normalization layer for a convolution in a neural network
    """

    gamma: Tensor
    beta: Tensor
    mu: np.ndarray
    sigma: np.ndarray
    mu_avg: np.ndarray
    sigma_avg: np.ndarray
    momentum: float
    x_hat: Tensor

    def __init__(self, channels: int, momentum: float = 0.9):
        super().__init__()
        self.gamma = Tensor(np.random.randn(channels, 1))
        self.beta = Tensor(np.zeros(shape=(channels, 1)))
        self.mu_avg = np.zeros_like(self.gamma)
        self.sigma_avg = np.ones_like(self.gamma)
        self.momentum = momentum

    def forward(self, inp: Tensor):
        """
        Forward assuming input is in form (n, c, p, p)
        """
        self.inp = inp
        temp = np.transpose(inp.value, axes=(1, 0, 2, 3))
        temp = np.reshape(temp, shape=(temp.shape[0], -1))

        if self.training:
            self.mu = np.mean(temp, axis=1, keepdims=True)
            nu = np.var(temp, axis=1, keepdims=True)
            self.sigma = np.sqrt(1e-8 + nu)

            self.mu_avg = self.momentum * self.mu_avg + (1 - self.momentum) * self.mu
            self.sigma_avg = self.momentum * self.sigma_avg + (1 - self.momentum) * self.sigma

            self.x_hat = Tensor((temp - self.mu) / self.sigma) # Normalize
            self.out = Tensor(np.reshape(self.gamma.value * self.x_hat.value + self.beta.value, shape=inp.shape))
        else:
            norm = (temp - self.mu_avg) / self.sigma_avg
            self.out = Tensor(np.reshape(self.gamma.value * norm + self.beta.value, inp.shape)) # Reparametrize

    def backward(self):
        """
        Backward pass, assuming out grad is populated
        """
        out_grad = np.transpose(self.out.grad, axes=(1, 0, 2, 3))
        out_grad = np.reshape(out_grad, shape=self.x_hat.shape)

        self.gamma.grad = np.sum(out_grad * self.x_hat.value, axis=1, keepdims=True)
        self.beta.grad = np.sum(out_grad, axis=1, keepdims=True)
        self.x_hat.grad = self.gamma.value * out_grad

        dx = 1.0 / (self.sigma * out_grad.shape[1]) * (out_grad.shape[1] *
            self.x_hat.grad - np.sum(self.x_hat.grad, axis=1, keepdims=True) -
            self.x_hat.value * np.sum(self.x_hat.grad * self.x_hat.value, axis=1, keepdims=True))

        self.inp.grad = np.reshape(dx,
                    (self.inp.shape[1], self.inp.shape[0], self.inp.shape[2], self.inp.shape[3]))
        self.inp.grad = self.inp.grad.transpose(1, 0, 2, 3)

    def parameters(self):
        """ Return the batch norm layer parameters"""
        return [self.gamma, self.beta]

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
            self.sigma = np.sqrt(1e-5 + nu)

            self.mu_avg = self.momentum * self.mu_avg + (1 - self.momentum) * self.mu
            self.sigma_avg = self.momentum * self.sigma_avg + (1 - self.momentum) * self.sigma

            self.x_hat = Tensor((temp - self.mu) / self.sigma) # Normalize

            out = self.gamma.value * self.x_hat.value + self.beta.value
        else:
            norm = (temp - self.mu_avg) / np.sqrt(self.sigma_avg**2 + 1e-5)
            out = self.gamma.value * norm + self.beta.value # Reparametrize

        out = np.reshape(out, (inp.shape[1], inp.shape[0], inp.shape[2], inp.shape[3]))
        out = np.transpose(out, axes=(1, 0, 2, 3))
        self.out = Tensor(out)

    def backward(self):
        """
        Backward pass, assuming out grad is populated
        """
        out_grad = np.transpose(self.out.grad, axes=(1, 0, 2, 3))
        out_grad = np.reshape(out_grad, shape=self.x_hat.shape)
        m = out_grad.shape[1]

        self.gamma.grad = np.sum(out_grad * self.x_hat.value, axis=1, keepdims=True)
        self.beta.grad = np.sum(out_grad, axis=1, keepdims=True)
    
        self.x_hat.grad = out_grad * self.gamma.value

        sum_dxhat = np.sum(self.x_hat.grad, axis=1, keepdims=True)
        sum_dxhat_xhat = np.sum(self.x_hat.grad * self.x_hat.value, axis=1, keepdims=True)

        dx = (1. / (self.sigma * m)) * (m * self.x_hat.grad - sum_dxhat - self.x_hat.value * sum_dxhat_xhat)
        self.inp.grad = dx.reshape(self.inp.shape[1], self.inp.shape[0], self.inp.shape[2], self.inp.shape[3]).transpose(1, 0, 2, 3)


    def parameters(self):
        """ Return the batch norm layer parameters"""
        return [self.gamma, self.beta]

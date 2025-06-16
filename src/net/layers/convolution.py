"""
File containing the convolutional layer class
"""

import numpy as np
from net.tensor import Tensor
from net.layer import Layer
from net.util import im2col, col2im

class Convolutional(Layer):
    """
    A class representing a convolutional layer in a neural network
    """
    kernel: Tensor
    bias: Tensor
    strides: int
    padding: int
    cols: np.ndarray
    out_shape: tuple

    def __init__(self, kernel: Tensor, strides: int = 1, padding: int = 1):
        super().__init__()
        self.kernel = kernel
        self.bias = Tensor(np.zeros((kernel.shape[0], 1)))
        self.kernel.reshape((kernel.shape[0], -1)) # For fast convolutions
        self.strides = strides
        self.padding = padding

    def forward(self, inp: Tensor) -> None:
        """
        A forward pass through a convolutional layer using
        strided view for efficient operations
        """

        # Alias input
        self.inp = inp

        # Get kernel size
        kh, kw = self.kernel.shape_hint[2], self.kernel.shape_hint[3]

        # Create view of input for fast convolution
        view, out_h, out_w = im2col(inp.value, (kh, kw), self.strides, self.padding)

        # Cache for use during backprop
        self.cols = view
        self.out_shape = (out_h, out_w)

        # Unshaped output
        out_unshaped = self.kernel.value @ view
        out_unshaped += self.bias.value

        # Unbind dimensions of output
        self.out = Tensor(np.reshape(out_unshaped,
                                     shape=(inp.shape[0], self.kernel.shape[0], out_h, out_w)))

    def backward(self) -> None:
        """
        Apply back propagation on network assuming back propagated has populated out.grad
        """
        # Transpose out grad
        out_grad = np.transpose(self.out.grad, axes=(1, 0, 2, 3))

        # Reshape out grad
        out_grad = np.reshape(out_grad, (out_grad.shape[0], -1))
        self.bias.grad = np.sum(out_grad, axis=1, keepdims=True)

        # Get kernel gradient
        self.kernel.grad = out_grad @ self.cols.T

        # First get column gradient then turn to im shape
        col_grad = self.kernel.value.T @ out_grad

        # Get kernel size
        kh, kw = self.kernel.shape_hint[2], self.kernel.shape_hint[3]

        # Apply col2im then set input grad
        self.inp.grad = col2im(
            cols=col_grad,
            input_shape=self.inp.shape,
            kernel_size=(kh, kw),
            stride=self.strides,
            padding=self.padding,
            output_shape=self.out_shape
        )

    def parameters(self) -> list[Tensor]:
        """
        Return a singleton list containing a kernel
        """
        return [self.kernel, self.bias]

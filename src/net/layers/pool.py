"""
File containing the pooling layer
"""

import numpy as np
from net.layer import Layer
from net.tensor import Tensor

class Pool(Layer):
    """
    Class for the pooling layer, defaulting to max pooling
    """

    size: int
    stride: int
    mask: np.ndarray

    def __init__(self, size: int, stride: int):
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self, inp: Tensor) -> None:
        """
        Forward pass through the pooling layer
        """
        self.inp = inp
        n, c, h, w = inp.shape
        out_h = (h - self.size) // self.stride + 1
        out_w = (w - self.size) // self.stride + 1

        # For back pass
        self.mask = np.zeros_like(inp.value)

        # Construct out vals
        self.out = Tensor(np.zeros(shape=(n, c, out_h, out_w)))

        for i in range(out_h):
            for j in range(out_w):

                # set bounds
                h_start = i * self.stride
                h_end = h_start + self.size
                w_start = j * self.stride
                w_end = w_start + self.size

                pool = inp.value[:, :, h_start:h_end, w_start:w_end]
                max_val = np.max(pool, axis=(2, 3))
                self.out.value[:, :, i, j] = max_val

                # Save mask
                mask = (pool == max_val[:, :, None, None]) # For broadcasting
                self.mask[:, :, h_start:h_end, w_start:w_end] += mask
    
    def backward(self) -> None:
        """
        Run backward pass on pooling layer
        """
        out_h, out_w = self.out.shape[2], self.out.shape[3]

        for i in range(out_h):
            for j in range(out_w):

                # set bounds
                h_start = i * self.stride
                h_end = h_start + self.size
                w_start = j * self.stride
                w_end = w_start + self.size

                grad_vals = self.out.grad[:, :, i:i+1, j:j+1] # To allow for broadcasting below

                # Only flow gradient to max vals
                upd = grad_vals * (self.mask[:, :, h_start:h_end, w_start:w_end])
                self.inp.grad[:, :, h_start:h_end, w_start:w_end] += upd

    def parameters(self):
        """
        This has no parameters
        """ 
        return []
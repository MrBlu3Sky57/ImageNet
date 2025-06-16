"""
Module containing utility functions
"""

import numpy as np
from net.tensor import Tensor

def relu(x: Tensor):
    """
    ReLU activation function
    """
    return Tensor(np.maximum(0, x.value))

def d_relu(x: Tensor):
    """
    ReLU derivative
    """
    return (x.value > 0).astype(float)

def as_strided(*args, **kwargs):
    """
    A wrapper function for the NumPy as_strided function
    to avoid memory overflow
    """
    out = np.lib.stride_tricks.as_strided(*args, **kwargs)
    assert np.prod(out.shape) * out.itemsize <= args[0].nbytes, 'Memory Overflow Risk'
    return out

def im2col(inp: np.ndarray, kernel_size: tuple[int, int], stride: int = 1,
           padding: int = 0) -> tuple:
    """
    Return a view into the input array of the given size --- Need to change to sliding window view
    """
    n, c, h, w = inp.shape
    kh, kw = kernel_size
    if padding > 0:
        inp = np.pad(inp, ((0, 0), (0, 0), (padding ,padding), (padding, padding)))

    # Update shapes after padding
    n, c, h, w = inp.shape

    # Manually calculate output dimensions
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    strides = (
        inp.strides[0],
        inp.strides[1],
        stride * inp.strides[2],
        stride * inp.strides[3],
        inp.strides[2],
        inp.strides[3]
    )
    strided = as_strided(inp, shape=(n, c, out_h, out_w, kh, kw), strides=strides)

    # Reshape for matmul
    cols = np.transpose(strided, (c, kh, kw, n, out_h, out_w))
    cols = np.reshape(cols, (c * kh * kw, n * out_h * out_w))

    return cols, out_h, out_w

def col2im(cols: np.ndarray,
           input_shape: tuple[int, int, int, int],
           kernel_size: tuple[int, int],
           stride: int,
           padding: int,
           output_shape: tuple[int, int]) -> np.ndarray:
    """
    Converts columns back into the original image shape, reversing im2col.
    --- I did not want to write this, chat GPT did it based on my instruction ---
    """
    n, c, h, w = input_shape
    kh, kw = kernel_size
    out_h, out_w = output_shape

    # Reshape back to the sliding window view shape
    cols_reshaped = cols.reshape(c, kh, kw, n, out_h, out_w)
    cols_reshaped = cols_reshaped.transpose(3, 0, 1, 2, 4, 5)  # (N, C, KH, KW, out_h, out_w)

    # Initialize padded output
    h_padded, w_padded = h + 2 * padding, w + 2 * padding
    padded = np.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)

    # Fill in the gradients using strides
    for i in range(kh):
        for j in range(kw):
            padded[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] += cols_reshaped[:, :, i, j]

    # Remove padding
    if padding == 0:
        return padded
    return padded[:, :, padding:-padding, padding:-padding]

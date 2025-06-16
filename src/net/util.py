"""
Module containing utility functions
"""

import numpy as np
from net.tensor import Tensor

SEED = 42

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

def tanh(x: Tensor):
    """
    Tanh function
    """
    return Tensor(np.tanh(x.value))

def dtanh(x: Tensor):
    """
    Tanh derivative
    """
    return 1 - np.tanh(x.value) ** 2

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

    cols = np.lib.stride_tricks.sliding_window_view(inp, window_shape=(kh, kw), axis=(2, 3))
    cols = cols[:, :, ::stride, ::stride, :, :]

    # Reshape for matmul
    cols = np.transpose(cols, axes=(1, 4, 5, 0, 2, 3))
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
            padded[:, :, i:i + stride * out_h:stride, 
                   j:j + stride * out_w:stride] += cols_reshaped[:, :, i, j]

    # Remove padding
    if padding == 0:
        return padded
    return padded[:, :, padding:-padding, padding:-padding]

def cross_entropy(logits: np.ndarray, targets: np.ndarray, grad: bool = True) -> np.ndarray | tuple:
    """
    Cross entropy loss for given logits with respect to targets using softmax. Optionally
    returns gradient
    """
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    n = logits.shape[0]
    loss = np.mean(-np.log(probs[np.arange(n), targets]))

    if grad:
        grad = np.copy(probs)
        grad[np.arange(n), targets] -= 1 # softmax - y_onehot
        grad /= n

        return loss, grad
    else:
        return loss

def clip_grad(tensor: Tensor, max_norm: float):
    """ Clip grad """
    grad_norm = np.linalg.norm(tensor.grad)
    if grad_norm > max_norm:
        tensor.grad *= (max_norm / grad_norm)
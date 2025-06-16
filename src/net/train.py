"""
File containing the training function
"""

import numpy as np
from net.network import Network
from net.util import SEED, clip_grad

def grad_descent(model: Network, loss: callable, xs: np.ndarray, ys: np.ndarray, steps: int, batches: int, lr: float):
    """
    Run stochastic gradient descent on model

    --- Assuming model will apply softmax then cross entropy loss to outputs ---
    """

    generator = np.random.default_rng(SEED)
    for step in range(steps):
        idcs = generator.integers(0, len(xs), batches)
        x_batch, y_batch = xs[idcs], ys[idcs]

        # Run forward pass
        out = model.forward(x_batch)

        # Get loss grad
        loss_val, loss_grad = loss(out.value, y_batch)
        out.grad = loss_grad

        # Run backward pass
        model.backward()
        
        for p in model.parameters():
            clip_grad(p, 5.0)
            p.increment(lr)
        


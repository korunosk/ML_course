# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = y.shape[0]
    
    e = y - np.matmul(tx, w)
    
    return 1.0 / (2*N) * np.sum( e ** 2 )
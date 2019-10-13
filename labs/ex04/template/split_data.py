# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    sz = x.shape[0]
    idx = list(np.random.choice(sz, int(sz * ratio), replace=False))
    i_idx = list(set(range(sz)) - set(idx))
    return x[idx], y[idx], x[i_idx], y[i_idx]

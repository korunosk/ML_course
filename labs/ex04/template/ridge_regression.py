# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse


def ridge_regression(y, tx, lambda_):
    N, D = tx.shape
    w = np.linalg.inv(tx.T @ tx + 2 * N * lambda_ * np.identity(D)) @ tx.T @ y
    mse = compute_mse(y, tx, w)
    return w, mse

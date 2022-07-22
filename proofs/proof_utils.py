import numpy as np
from numpy import ndarray
from typing import Callable

def matrix_derivative_numerical(f: Callable[[ndarray], float], X: ndarray, dx=1e-4):
    """
    Numerically evaluate the derivative of the function f, which maps matrices to scalars, at a point X.
    Use a forward-backward method with step size dx.
    """

    out = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_ = X.copy()
            _X = X.copy()
            X_[i, j] += dx / 2
            _X[i, j] -= dx / 2
            out[i, j] = (f(X_) - f(_X)) / dx

    return out


def vector_derivative_numerical(f: Callable[[ndarray], float], x: ndarray, dx=1e-4):
    """
    Numerically evaluate the derivative of the function f, which maps vectors to scalars, at a point x.
    Use a forward-backward method with step size dx.
    """
    out = np.zeros_like(x)

    for i in range(len(x)):
        x_ = x.copy()
        _x = x.copy()
        x_[i] += dx / 2
        _x[i] -= dx / 2
        out[i] = (f(x_) - f(_x)) / dx

    return out
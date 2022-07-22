from numpy import ndarray
import numpy as np
from typing import Callable


def solve_SIM(y: ndarray,
              Minv: Callable,
              MinvN: Callable,
              tol: float=1e-5,
              verbose: bool=True,
              max_iter=20000):
    """
    Compute the Stationary Iterative Method solution to the linear equation Ax = y, where A
    has been split as A = M - N. To complete the algorithm, provide efficient implementations
    of the linear operators:

        * x -> M^{-1} x
        * x -> M^{-1} N x

    as callable functions. `tol` specifies
    """

    dx = Minv(y)
    x = np.zeros_like(dx)

    x += dx
    nits = 0

    dx_max = dx.max()
    while (dx_max > tol).any():

        dx = MinvN(dx)
        dx_max = dx.max()
        x += dx
        nits += 1

        # if verbose:
        #     print(f'dx = {dx_max:.4E} > {tol:.4E}', end='\r')

        if nits == max_iter:
            print('WANRING: MAXIMUM NUMBER OF ITERACTIONS REACHED. RESULT MAY BE INCCURATE.')
            break

    return x, nits


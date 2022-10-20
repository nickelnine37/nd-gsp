import numpy as np
from typing import Union
from numpy import ndarray
from scipy.sparse import spmatrix

from pykronecker.base import KroneckerOperator

from ndgsp.utils.types import Operator, Array


def solve_SPCGM(A_precon: Operator,
                y: Array,
                Phi: Operator,
                x0: Array = None,
                max_iter=20000,
                reltol=1e-8,
                verbose=False
                ) -> Array:
    """
    Solve a symmetric preconditioned version of the conjugate gradient method. Here, we look to
    solve the linear equation

    (Φ.T A Φ) (inv(Φ) x) = Φ.T y

    Parameters
    ----------
    A_precon        The preconditioned coefficient matrix, (Φ.T A Φ)
    y               The vector to solve against
    Phi             The right symmetric preconditioner Φ

    other arguments are as in solve_CMG

    Returns
    -------
    x               The result of solving the linear system
    """

    return Phi @ solve_CGM(A=A_precon, y=Phi.T @ y, x0=x0, max_iter=max_iter, reltol=reltol, verbose=verbose)


def solve_CGM(A: Operator,
              y: Array,
              x0: Array = None,
              max_iter=20000,
              reltol=1e-8,
              verbose=False) -> Array:
    """

    Use the conjugate gradient method to solve the linear system Ax = y.

    Parameters
    ----------
    A               The coefficient matrix, A
    y               The vector y in Ax = y
    x0              An optional initial guess for x
    max_iter        Maximum number of iterations
    reltol          The relative tolerance level
    verbose         If True prints some extra information

    Returns
    -------
    x               The result of solving the linear system
    """

    if x0 is None:
        x = np.zeros_like(y)

    else:
        x = x0

    r = y - A @ x

    d = r
    res_new = (r ** 2).sum()
    res0 = res_new

    its = 0

    while its < max_iter and res_new > (reltol ** 2 * res0):

        its += 1

        Ad = A @ d

        alpha = res_new / (d * Ad).sum()

        x += alpha * d

        # periodically rescale
        if its % 50 == 0:
            r = y - A @ x
            d = r

        else:
            r -= alpha * Ad

        res_old = res_new
        res_new = (r ** 2).sum()
        d = r + d * res_new / res_old

    if its == max_iter:
        print(f'Warning: failed to converge in {its} iterations')

    return x

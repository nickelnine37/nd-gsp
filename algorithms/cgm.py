import numpy as np
from typing import Callable, Union
from numpy import ndarray
from scipy.sparse import spmatrix

from utils.kronecker import KroneckerOperator


def solve_SPCGM(A_precon: Union[ndarray, KroneckerOperator, spmatrix],
                y: ndarray,
                x0: ndarray= None,
                Phi: Union[ndarray, KroneckerOperator, spmatrix] = None,
                PhiT: Union[ndarray, KroneckerOperator, spmatrix] = None,
                max_iter=20000,
                reltol=1e-8,
                verbose=False
                ) -> tuple[ndarray, int]:
    """
    Solve a symmetric preconditioned version of the conjugate gradient method. Here, we look to
    solve the linear equation

    (Φ.T A Φ) (inv(Φ) x) = Φ.T y

    Parameters
    ----------
    A_precon        The preconditioned coefficient matrix, (Φ.T A Φ)
    Phi             The right symmetric preconditioner Φ
    PhiT            The left symmetric preconditioner Φ.T

    other arguments are as in solve_CMG

    Returns
    -------
    x               The result of solving the linear system
    n_ints          The number of iterations completed

    """

    x_, its = solve_CMG(A=A_precon, y=PhiT @ y, x0=x0, max_iter=max_iter, reltol=reltol, verbose=verbose)
    return Phi @ x_, its


def solve_CMG(A: Union[ndarray, KroneckerOperator, spmatrix],
              y: ndarray,
              x0: ndarray = None,
              max_iter=20000,
              reltol=1e-8,
              verbose=False) -> tuple[ndarray, int]:
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
`
    Returns
    -------
    x               The result of solving the linear system
    n_ints          The number of iterations completed
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

    return x, its


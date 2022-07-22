import numpy as np
from typing import Callable

def solve_SPCGM(A_precon: Callable,
                y: np.ndarray,
                x0: np.ndarray = None,
                Phi: Callable[[np.ndarray], np.ndarray] = None,
                PhiT: Callable[[np.ndarray], np.ndarray] = None,
                max_iter=20000,
                reltol=1e-8,
                verbose=False
                ) -> tuple[np.ndarray, int]:
    """
    Solve a symmetric preconditioned version of the conjugate gradient method. Here, we look to
    solve the linear equation

    (Φ.T A Φ) (inv(Φ) x) = Φ.T y

    Parameters
    ----------
    A_precon        The preconditioned coefficient matrix, as a function which performs the operation x -> (Φ.T A Φ) x
    Phi             The right symmetric preconditioner, as a funnction x -> Φ x
    PhiT            The left symmetric preconditioner, as a funnction x -> Φ.T x

    other arguments are as in solve_CMG

    Returns
    -------
    x               The result of solving the linear system
    n_ints          The number of iterations completed

    """

    x_, its = solve_CMG(A=A_precon, y=PhiT(y), x0=x0, max_iter=max_iter, reltol=reltol, verbose=verbose)
    return Phi(x_), its


def solve_CMG(A: Callable[[np.ndarray], np.ndarray],
              y: np.ndarray,
              x0: np.ndarray = None,
              max_iter=20000,
              reltol=1e-8,
              verbose=False) -> tuple[np.ndarray, int]:
    """

    Use the conjugate gradient method to solve the linear system Ax = y.

    Parameters
    ----------
    A               The coefficient matrix, as a function which performs the operation x -> A x
    y               The vector y in Ax = y
    x0              An optional initial guess for x
    max_iter        Maximum number of iterations
    reltol          The relative tolerance level
    verbose         If True prints some extra information

    Returns
    -------
    x               The result of solving the linear system
    n_ints          The number of iterations completed
    """

    if x0 is None:
        x = np.zeros_like(y)
    else:
        x = x0

    r = y - A(x)

    d = r
    res_new = (r ** 2).sum()
    res0 = res_new

    its = 0

    while its < max_iter and res_new > (reltol ** 2 * res0):

        its += 1

        Ad = A(d)

        alpha = res_new / (d * Ad).sum()

        x += alpha * d

        # periodically rescale
        if its % 50 == 0:
            r = y - A(x)
            d = r

        else:
            r -= alpha * Ad

        res_old = res_new
        res_new = (r ** 2).sum()
        d = r + d * res_new / res_old

    if verbose:
        if its == max_iter:
            print(f'Warning: failed to converge in {its} iterations')
        else:
            print(f'Completed in {its} iterations')

    return x, its


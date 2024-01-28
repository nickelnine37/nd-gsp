import numpy as np
from typing import Tuple
# from scipy.sparse.linalg import cg, LinearOperator

from jax.scipy.sparse.linalg import cg

from ndgsp.utils.types import Operator, Array


def solve_SPCGM(A_precon: Operator,
                Y: Array,
                Phi: Operator,
                max_iter=20000,
                tol=None,
                ) -> Tuple[Array, int]:
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
    out, nits = solve_CGM(A=A_precon, Y=Phi.T @ Y, max_iter=max_iter, tol=tol)

    # print(type(out), out.shape)

    # print(type(Phi), Phi.shape)

    return Phi @ out, nits


def solve_CGM(A: Operator,
              Y: Array,
              max_iter=10000,
              tol=1e-8) -> Tuple[Array, int]:
    """

    Use the conjugate gradient method to solve the linear system Ax = y.

    Parameters
    ----------
    A               The coefficient matrix, A
    y               The vector y in Ax = y
    x0              An optional initial guess for x
    max_iter        Maximum number of iterations
    reltol          The relative tolerance level

    Returns
    -------
    x               The result of solving the linear system
    """

    n_elements = np.prod(Y.shape)
    nits = 0

    def iter_count(arr):
        nonlocal nits
        nits += 1

    # linop = LinearOperator((n_elements, n_elements), matvec=lambda x: A @ x)
    # z, exit_code = cg(linop, Y.ravel(), callback=iter_count, maxiter=max_iter)

    z, exit_code = cg(A, Y, maxiter=max_iter)
 
    return z.reshape(Y.shape), nits



def solve_SIM(Minv: Operator,
              N: Operator,
              Y: Array,
              max_iters: int = 10000,
              tol: float = 1e-8) -> Tuple[Array, int]:
    """
    Solve a linear system using the stationary iterative method of matrix splitting. In particular,
    we wish to solve the system f = inv(A) @ y by splitting the coefficient matrix A into A = M - N.

    Params:
        Minv:       inv(M) as an operator. Can be an array or a KroneckerOperator.
        N:          N as an operator. Can be an array or a KroneckerOperator.
        Y:          Y in the linear system
        max_iters:  The maximum number of iterations
        tol:        The convergence tolerance

    Returns:
        F:          The solution to the linear system
        nits:       The number of iterations taken to converge
    """

    n_elements = np.prod(Y.shape)
    dF = Minv @ Y
    F = dF
    nits = 0

    while (dF ** 2).sum() ** 0.5 / n_elements > tol:

        dF = Minv @ N @ dF
        F += dF
        nits += 1

        if nits == max_iters:
            print(f'Warning: Maximum iterations ({max_iters}) reached')
            break

    return F, nits
import numpy as np
from typing import Union, Tuple
from numpy import ndarray
from scipy.sparse import spmatrix

from pykronecker.base import KroneckerOperator
from scipy.sparse.linalg import cg, LinearOperator

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

    linop = LinearOperator((n_elements, n_elements), matvec=lambda x: A @ x)
    z, exit_code = cg(linop, Y.ravel(), callback=iter_count, maxiter=max_iter)

    return z.reshape(Y.shape), nits


if __name__ == '__main__':

    from ndgsp.graph.graphs import ProductGraph
    from ndgsp.graph.filters import MultivariateFilterFunction
    from pykronecker import KroneckerDiag, KroneckerIdentity

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    np.random.seed(0)

    N = 10
    T = 20

    graph = ProductGraph.lattice(N, T)
    f_func = MultivariateFilterFunction.diffusion([0.2, 0.2])
    gamma = 1
    S = np.random.randint(2, size=(N, T))
    DS = KroneckerDiag(S)
    Y = np.random.normal(size=(N, T))

    G = f_func(graph.lams)
    DG = KroneckerDiag(G)

    A_precon = DG @ graph.U.T @ DS @ graph.U @ DG + gamma * KroneckerIdentity(like=DS)
    Phi = graph.U @ DG

    F, nits = solve_SPCGM(A_precon, Y, Phi)

    Hi2 = (graph.U @ KroneckerDiag(G ** -2) @ graph.U.T).to_array()
    F_ = np.linalg.solve(np.diag(S.reshape(-1)) + gamma * Hi2, Y.reshape(-1)).reshape(N, T)

    print(np.allclose(F, F_, atol=1e-4))

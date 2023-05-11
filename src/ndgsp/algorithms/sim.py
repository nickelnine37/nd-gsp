from ndgsp.utils.types import Operator, Array
from typing import Tuple
import numpy as np


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


if __name__ == '__main__':

    from ndgsp.graph.graphs import ProductGraph
    from ndgsp.graph.filters import MultivariateFilterFunction
    from pykronecker import KroneckerDiag

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    np.random.seed(0)

    N = 10
    T = 20

    graph = ProductGraph.lattice(N, T)
    f_func = MultivariateFilterFunction.diffusion([0.2, 0.2])
    gamma = 1
    S = np.random.randint(2, size=(N, T))
    Y = np.random.normal(size=(N, T))

    G = f_func(graph.lams)
    J = (G ** 2) / (G ** 2 + gamma)

    Minv = graph.U @ KroneckerDiag(J) @ graph.U.T

    F, nits = solve_SIM(Minv, KroneckerDiag(1 - S), Y)
    Hi2 = (graph.U @ KroneckerDiag(G ** -2) @ graph.U.T).to_array()
    F_ = np.linalg.solve(np.diag(S.reshape(-1)) + gamma * Hi2, Y.reshape(-1)).reshape(N, T)

    print(np.allclose(F, F_, atol=1e-4))




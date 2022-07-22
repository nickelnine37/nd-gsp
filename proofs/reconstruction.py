import numpy as np
from numpy import trace as tr, eye as I, kron, diag
from numpy.linalg import inv, solve
from numpy.random import randn, randint

from models.reconstruction import SignalProjector, get_y_and_s
from utils.linalg import vec, ten
from graph.graphs import Graph
from graph.filters import UnivariateFilterFunction
from proof_utils import vector_derivative_numerical, matrix_derivative_numerical


def test1():
    """
    For a cost function

        $$
        C(f) = ||y - s · f||^2 + ɣ * f.T H^-2 f
        $$

    The minimising value of $F$ is

        $$
        ( diag(s) + ɣ H^-2 )^-1 y
        $$

    which is equal to
        $$
        H^2( H^2 diag(s) + ɣ I)^-1 y
        $$

    """

    N = 10

    graph = Graph.chain(N)
    filter_function = UnivariateFilterFunction.diffusion(beta=1)
    gamma = randn() ** 2

    y = randn(N)
    s = randint(0, 2, N)
    y[~s.astype(bool)] = 0

    Hi = graph.U @ diag(filter_function(graph.lam) ** -2) @ graph.U.T
    H = graph.U @ diag(filter_function(graph.lam) ** 2) @ graph.U.T

    def nll(f):
        return ((y - s * f).T @ (y - s * f)) + gamma * f @ Hi @ f

    f1 = inv(diag(s) + gamma * Hi) @ y
    f2 = H @ inv(diag(s) @ H + gamma * I(N)) @ y

    assert np.allclose(vector_derivative_numerical(nll, f1), 0, atol=1e-5)
    assert np.allclose(f1, f2)


def test2():
    """
    The same result can be achieved as above by inverting a down-projected version.
    """

    N = 10

    graph = Graph.chain(N)
    filter_function = UnivariateFilterFunction.diffusion(beta=1)
    gamma = randn() ** 2

    signal = np.random.randn(N)
    signal[np.random.randint(0, 2, N)] = np.nan
    projector = SignalProjector(signal)
    y, s = get_y_and_s(signal)

    H = graph.U @ diag(filter_function(graph.lam) ** 2) @ graph.U.T
    f1 = H @ inv(diag(s) @ H + gamma * I(N)) @ y

    y_ = projector.down_project_signal(y)
    M = projector.down_project_operator(H) + gamma * I(projector.N_)
    f2 = H @ projector.up_project_signal(solve(M, y_))

    print(f1)

    assert np.allclose(f1, f2)



if __name__ == '__main__':

    test1()
    test2()

    print('All tests passed')
from graph.graphs import BaseGraph, Graph, ProductGraph
from graph.signals import PartiallyObservedGraphSignal, PartiallyObservedProductGraphSignal
from graph.filters import FilterFunction, UnivariateFilterFunction, MultivariateFilterFunction
from utils.checks import check_valid_graph, check_compatible

from typing import Union
import numpy as np
from numpy import eye as I, diag, ndarray
import networkx as nx


class SignalProjector:

    def __init__(self, s: ndarray):
        """
        s is a boolean array specifying where measurements were made
        """

        self.s = s
        self.N = len(s)
        self.N_ = s.sum()

    def down_project_signal(self, f: ndarray):
        """
        Condense a vector f such that it contains only elements specified by s
        f (N, ) -> f_ (N_, )
        """
        assert len(f) == self.N
        return f[self.s]

    def up_project_signal(self, f_: ndarray):
        """
        Upsample f so that it is padded with zeros where no observation was made
        f_ (N_, ) -> f (N, )
        """
        assert len(f_) == self.N_
        f = np.zeros(self.N)
        f[self.s] = f_
        return f

    def down_project_operator(self, A: ndarray):
        """
        Condense a matrix A, removing rows and columns as specified by s
        A (N, N) -> A_ (N_, N_)
        """
        assert A.shape == (self.N, self.N), f'passed array should have shape {(self.N, self.N)} but it has shape {A.shape}'
        return A[:, self.s][self.s, :]

    def up_project_operator(self, A_: ndarray):
        """
        Upsample a matrix A_, adding zeros columns and rows appropriately
        A_ (N_, N_) -> A (N, N)
        """
        assert A_.shape == (self.N_, self.N_)
        A = np.zeros(self.N ** 2)
        A[(self.s[:, None] * self.s[None, :]).reshape(-1)] = A_.reshape(-1)
        return A.reshape(self.N, self.N)


def get_y_and_s(signal: ndarray):
    """
    Take in a user-provided partially observed graph signal. This can be either:

        * an array containing nans to indicate missing data,
        * an np.ma.MaskedArray object

    Return this array with the missing values filled with zeros, and a boolean array
    of the same shape holding True where observations were made.

    """
    if isinstance(signal, np.ma.MaskedArray):
        s = ~signal.mask
        y = signal.data

    elif isinstance(signal, ndarray):
        s = ~np.isnan(signal)
        y = signal

    else:
        raise TypeError('signal should be an array or a masked array')

    y[~s] = 0

    return y, s


def solve_SIM(Y: ndarray, S_: ndarray, J: ndarray, graph: BaseGraph, tol: float=1e-5):

    assert Y.shape == S_.shape and S_.shape == J.shape, f'Y, S_ and J should be the same shape but they are {Y.shape}, {S_.shape} and {J.shape} respectively'

    dF = graph.scale_spectral(Y, J)
    F = np.zeros_like(dF)

    F += dF

    while any(dF > tol):

        dF = graph.scale_spectral(S_ * dF, J)
        F += dF

    return F





class UnivariateGraphSignalReconstructor:

    def __init__(self,
                 signal: ndarray,
                 graph: Union[BaseGraph, ndarray, nx.Graph],
                 filter_function: FilterFunction,
                 gamma: float):


        self.y, s = get_y_and_s(signal)
        self.projector = SignalProjector(s)

        # validate the graph and turn into a graph.Graph if not already
        self.graph = check_valid_graph(graph)
        self.filter_function = filter_function
        self.gamma = gamma

        # check the signal, graph and filter_function are all mutually compatible
        check_compatible(signal=self.y, graph=self.graph, filter_function=self.filter_function)

        self.g2 = self.filter_function(self.graph.lam) ** 2
        self.H2 = self.graph.U @ diag(self.g2) @ self.graph.U.T

        self.M = self.projector.down_project_operator(self.H2) + self.gamma * I(self.projector.N_)

    def set_gamma(self, gamma: float) -> 'UnivariateGraphSignalReconstructor':
        """
        Set the gamma parameter. Recompute M only.
        """
        self.gamma = gamma
        self.M = self.projector.down_project_operator(self.H2) + self.gamma * I(self.projector.N_)
        return self

    def set_beta(self, beta: Union[float, ndarray]) -> 'UnivariateGraphSignalReconstructor':
        """
        Set the beta parameter for the filter function. Recompute g2, H2 and M.
        """
        self.filter_function.set_beta(beta)
        self.g2 = self.filter_function(self.graph.lam) ** 2
        self.H2 = self.graph.scale_spectral(self.y, self.g2)
        self.M = self.projector.down_project_operator(self.H2) + self.gamma * I(self.projector.N_)

        return self

    def compute_mean(self) -> ndarray:
        """
        Compute the reconstructed signal. `signal` should be the appropriate shape for the graph. For indicating which
        pixels should be treated as missing, either:

            * pass an array containing nans to indicate missing data,
            * pass a np.ma.MaskedArray object

        """
        y_ = self.projector.down_project_signal(self.y)
        return self.H2 @ self.projector.up_project_signal(np.linalg.solve(self.M, y_))

    def compute_var(self) -> ndarray:
        """
        Compute the marginal variance associted with the prediction
        """
        return diag(self.compute_covar())


    def compute_covar(self) -> ndarray:
        return self.H2 @ self.projector.up_project_operator(np.linalg.inv(self.M))


    def compute_posterior(self):
        """
        Calling the class on a signal will compute both the posterior mean and the posterior variance
        """

        covar = self.compute_covar()
        mean = covar @ self.projector.down_project_signal(self.y)
        return mean, diag(covar)


#
# class MultivariateGraphSignalReconstructor:
#
#     def __init__(self,
#                  signal: ndarray,
#                  graph: Union[BaseGraph, ndarray, nx.Graph],
#                  filter_function: FilterFunction,
#                  gamma: float):
#
#         self.Y, self.S = get_y_and_s(signal)
#
#
#         # validate the graph and turn into a graph.Graph if not already
#         self.graph = check_valid_graph(graph)
#         self.filter_function = filter_function
#         self.gamma = gamma
#
#         # check the signal, graph and filter_function are all mutually compatible
#         check_compatible(signal=self.y, graph=self.graph, filter_function=self.filter_function)
#


if __name__ == '__main__':

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    def run_tests():

        def test_univariate():

            # set variables
            N = 10
            graph = Graph.random_tree(N)
            signal = np.random.randn(N)
            s = np.random.randint(0, 2, N)
            gamma = 1
            filter_function = UnivariateFilterFunction.diffusion(beta=1)

            # calculate the solution explicitly
            H2 = graph.U @ diag(filter_function(graph.lam) ** 2) @ graph.U.T
            explicit_sigma = np.linalg.inv(np.linalg.inv(H2) + gamma * I(N))
            explicit_mean = explicit_sigma @ signal

            # use our special class
            reconstructor = UnivariateGraphSignalReconstructor(signal, graph, filter_function, gamma)
            reconstructor_mean, reconstructor_var = reconstructor.compute_posterior()

            assert np.allclose(explicit_mean, reconstructor_mean)
            assert np.allclose(diag(explicit_sigma), reconstructor_var)

        def test_multivariate():

            pass



        test_univariate()

        print('All tests passed')

    run_tests()






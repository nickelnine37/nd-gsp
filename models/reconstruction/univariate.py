from graph.graphs import BaseGraph, Graph
from graph.filters import FilterFunction, UnivariateFilterFunction
from models.reconstruction.reconstruction_utils import get_y_and_s
from utils.checks import check_valid_graph, check_compatible

from typing import Union
import numpy as np
from numpy import eye as I, diag, ndarray
import networkx as nx



class SignalProjector:

    def __init__(self, signal: ndarray):
        """
        s is a boolean array specifying where measurements were made
        """

        _, s = get_y_and_s(signal)

        self.s = s.astype(bool)
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


class UnivariateGraphSignalReconstructor:

    def __init__(self,
                 signal: ndarray,
                 graph: Union[BaseGraph, ndarray, nx.Graph],
                 filter_function: FilterFunction,
                 gamma: float):


        self.y, self.s = get_y_and_s(signal)
        self.projector = SignalProjector(signal)

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
        self.M = self.gamma * self.projector.down_project_operator(self.H2) + I(self.projector.N_)
        return self

    def set_beta(self, beta: Union[float, ndarray]) -> 'UnivariateGraphSignalReconstructor':
        """
        Set the beta parameter for the filter function. Recompute g2, H2 and M.
        """
        self.filter_function.set_beta(beta)
        self.g2 = self.filter_function(self.graph.lam) ** 2
        self.H2 = self.graph.U @ diag(self.g2) @ self.graph.U.T
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
        return self.H2 @ np.linalg.inv(diag(self.s) @ self.H2 + self.gamma * I(len(self.H2)))

    def compute_posterior(self):
        """
        Calling the class on a signal will compute both the posterior mean and the posterior variance
        """

        covar = self.compute_covar()
        mean = covar @ self.y
        return mean, diag(covar)


if __name__ == '__main__':

    def run_tests():

        # set variables
        N = 10
        graph = Graph.random_tree(N)
        signal = np.random.randn(N)
        signal[np.random.randint(0, 2, N)] = np.nan

        y, s = get_y_and_s(signal)

        gamma = np.random.randn() ** 2
        filter_function = UnivariateFilterFunction.diffusion(beta=1)

        # calculate the solution explicitly
        H2 = graph.U @ diag(filter_function(graph.lam) ** 2) @ graph.U.T

        def test_mean():

            explicit_mean = np.linalg.solve(gamma * np.linalg.inv(H2) + diag(s), y)
            reconstructor = UnivariateGraphSignalReconstructor(signal, graph, filter_function, gamma)
            reconstructor_mean = reconstructor.compute_mean()

            assert np.allclose(explicit_mean, reconstructor_mean)

        def test_var():

            explicit_var = diag(np.linalg.inv(gamma * np.linalg.inv(H2) + diag(s)))
            reconstructor = UnivariateGraphSignalReconstructor(signal, graph, filter_function, gamma)
            reconstructor_var = reconstructor.compute_var()

            assert np.allclose(explicit_var, reconstructor_var)

        test_mean()
        test_var()

        print('All tests passed')

    run_tests()

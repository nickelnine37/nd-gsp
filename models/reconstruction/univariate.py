from graph.graphs import BaseGraph, Graph
from graph.filters import _FilterFunction, UnivariateFilterFunction
from models.reconstruction.reconstruction_utils import get_y_and_s, SignalProjector
from utils.checks import check_valid_graph, check_compatible

from typing import Union
import numpy as np
from numpy import eye as I, diag, ndarray
import networkx as nx






class UnivariateGraphSignalReconstructor:

    def __init__(self,
                 signal: ndarray,
                 graph: Union[BaseGraph, ndarray, nx.Graph],
                 filter_function: _FilterFunction,
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

    def compute_logvar(self) -> ndarray:
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

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)


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

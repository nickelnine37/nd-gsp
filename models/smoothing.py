from graph.graphs import BaseGraph
from graph.filters import FilterFunction, MultivariateFilterFunction
from utils.checks import check_valid_graph, check_compatible
from typing import Union
from numpy import ndarray
import numpy as np
import networkx as nx


def smooth_graph_signal(signal: ndarray,
                        graph: Union[BaseGraph, ndarray, nx.Graph],
                        filter_function: FilterFunction,
                        gamma: float) -> tuple[ndarray, ndarray]:
    """
    Smooth a graph signal according to the Bayesian model. This is a wrapper for the class GraphSignalSmoother,
    to make a simpler API.

    Parameters
    ----------
    signal                   A graph signal with no missing values. This is an ndarray of any shape. If
    graph                    Either a Graph object, a graph adjacency matrix as a square ndarray (or spmatrix)
    filter_function          A FilterFunction object
    gamma                    Regularisation parameter (>0)

    Returns
    -------
    signal_                 The smoothed graph signal
    signal_var              The variance associated with the prediction
    """

    signal_smoother = GraphSignalSmoother(graph, filter_function, gamma)
    return signal_smoother(signal)


class GraphSignalSmoother:

    def __init__(self,
                 graph: Union[BaseGraph, ndarray, nx.Graph],
                 filter_function: FilterFunction,
                 gamma: float):
        """
        Create an object for performing graph signal smoothing

        Parameters
        ----------
        graph                    Either a Graph object, a graph adjacency matrix as a square ndarray (or spmatrix)
        filter_function          A FilterFunction object
        gamma                    Regularisation parameter (>0)

        """

        # validate the graph and turn into a graph.Graph if not already
        self.graph = check_valid_graph(graph)
        self.filter_function = filter_function
        self.gamma = gamma

        # check the signal, graph and filter_function are all mutually compatible
        check_compatible(graph=self.graph, filter_function=self.filter_function)

        # eigen-decompose the graph
        self.graph._decompose()

        # apply the filter function to the graph frequency
        if isinstance(filter_function, MultivariateFilterFunction):
            self.G2 = filter_function(graph.lams) ** 2
        else:
            self.G2 = filter_function(graph.lam) ** 2

        self.J = self.G2 / (self.G2 + gamma)

    def set_gamma(self, gamma: float) -> 'GraphSignalSmoother':
        """
        Set the gamma parameter. Recompute J only.
        """
        self.gamma = gamma
        self.J = self.G2 / (self.G2 + gamma)
        return self

    def set_beta(self, beta: Union[float, ndarray]) -> 'GraphSignalSmoother':
        """
        Set the beta parameter for the filter function. Recompute G2 and J.
        """
        self.filter_function.set_beta(beta)

        if isinstance(self.filter_function, MultivariateFilterFunction):
            self.G2 = self.filter_function(self.graph.lams) ** 2
        else:
            self.G2 = self.filter_function(self.graph.lam) ** 2

        self.J = self.G2 / (self.G2 + self.gamma)

        return self

    def compute_mean(self, signal: ndarray) -> ndarray:
        """
        Compute the smoothed signal
        """
        return self.graph.scale_spectral(signal, self.J)

    def compute_var(self) -> ndarray:
        """
        Compute the marginal variance associted with the prediction
        """
        return (self.graph.U ** 2) @ self.J

    def __call__(self, signal: ndarray):
        """
        Calling the class on a signal will compute both the posterior mean and the posterior variance
        """

        check_compatible(signal=signal, graph=self.graph, filter_function=self.filter_function)

        # check the signal contains no nans
        assert not np.isnan((signal ** 2).sum()), 'The signal contains nan values - this is not valid for graph signal smoothing'

        return self.compute_mean(signal), self.compute_var()



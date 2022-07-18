from graph.graphs import Graph, ProductGraph, BaseGraph
from graph.signals import GraphSignal, ProductGraphSignal
from graph.filters import FilterFunction, MultivariateFilterFunction
from utils.gsp import check_valid_graph
from typing import Union
from numpy import ndarray
import numpy as np
import networkx as nx


def smooth_graph_signal(signal: ndarray, graph: Union[BaseGraph, ndarray, nx.Graph], filter_function: FilterFunction, gamma: float) -> ndarray:
    """
    Smooth a graph signal according to the Bayesian model

    Parameters
    ----------
    signal          A graph signal with no missing values. This is an ndarray of any shape. If
    graph           Either a Graph object, a graph adjacency matrix as a square ndarray (or spmatrix)
    filter          A GraphFilter or SpaceTimeGraphFilter object
    gamma

    Returns
    -------

    """



    if isinstance(filter_function, MultivariateFilterFunction):
        assert isinstance(graph, ProductGraph), f'Space-time filters can only be used with product graphs, but it is type {type(graph)}'
        assert filter_function.n == graph.ndim, 'The dimension of the space-time filter and the product graph do not match'
        G2 = filter_function(graph.lams) ** 2

    else:
        G2 = filter_function(graph.lam) ** 2

    J = G2 / (G2 + gamma)

    return graph.scale_spectral(signal, J)


class GraphSignalSmoother:

    def __init__(self, signal: ndarray, graph: Union[BaseGraph, ndarray, nx.Graph], filter_function: FilterFunction, gamma: float):

        self.signal = signal

        self.graph = check_valid_graph(graph)
        self.graph.decompose()

        self.filter_function = filter_function
        self.gamma = gamma

        if isinstance(filter_function, MultivariateFilterFunction):
            assert isinstance(graph, ProductGraph), f'Space-time filters can only be used with product graphs, but it is type {type(graph)}'
            assert filter_function.n == graph.ndim, 'The dimension of the space-time filter and the product graph do not match'
            self.G2 = filter_function(graph.lams) ** 2

        else:
            self.G2 = filter_function(graph.lam) ** 2

        self.J = self.G2 / (self.G2 + gamma)

    def set_gamma(self, gamma):
        self.gamma = gamma
        self.J = self.G2 / (self.G2 + gamma)

    def set_beta(self, beta):
        self.filter_function.set_beta(beta)

        if isinstance(self.filter_function, MultivariateFilterFunction):
            self.G2 = self.filter_function(self.graph.lams) ** 2
        else:
            self.G2 = self.filter_function(self.graph.lam) ** 2

    def set_signal(self, signal):
        self.signal = signal

    def compute(self):
        return self.graph.scale_spectral(self.signal, self.J)



    



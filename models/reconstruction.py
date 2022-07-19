from graph.graphs import BaseGraph, Graph, ProductGraph
from graph.signals import PartiallyObservedGraphSignal, PartiallyObservedProductGraphSignal
from graph.filters import FilterFunction, UnivariateFilterFunction, MultivariateFilterFunction
from utils.gsp import check_valid_graph, check_compatible

from typing import Union
from numpy import eye as I, diag, ndarray
from numpy.linalg import solve, inv
import networkx as nx


class GraphSignalReconstructor:

    def __init__(self, signal: ndarray,
                        graph: Union[BaseGraph, ndarray, nx.Graph],
                        filter_function: FilterFunction,
                        gamma: float):

        # validate the graph and turn into a graph.Graph if not already
        self.graph = check_valid_graph(graph)

        self.signal = signal
        self.filter_function = filter_function
        self.gamma = gamma

        # check the signal, graph and filter_function are all mutually compatible
        check_compatible(self.signal, self.graph, self.filter_function)

        self.graph.decompose()

    def get_f(self):
        H2 = self.graph.U @ diag(self.filter_function(self.graph.lam) ** 2) @ self.graph.U.T
        y_ = self.signal.down_project_signal(self.signal.y)
        M = self.signal.down_project_operator(H2) + self.gamma * I(self.signal.N_)
        return H2 @ self.signal.up_project_signal(solve(M, y_))

    def get_sig(self):
        H2 = self.graph.U @ diag(self.filter_function(self.graph.lam) ** 2) @ self.graph.U.T
        M = self.signal.down_project_operator(H2) + self.gamma * I(self.signal.N_)
        return H2 @ self.signal.up_project_operator(inv(M))

    def solve(self):
        return self.get_f(), self.get_sig()


class ProductGraphSignalReconstructor:

    def __init__(self, graph: ProductGraph, signal: PartiallyObservedProductGraphSignal, filter: Union[UnivariateFilterFunction, MultivariateFilterFunction], gamma: float):

        self.graph = graph
        self.signal = signal
        self.filter = filter
        self.gamma = gamma

    def get_f(self, method='cgm'):

        if isinstance(self.filter, MultivariateFilterFunction):
            G2 = self.filter(self.graph.LamN, self.graph.LamT) ** 2
        else:
            G2 = self.filter(self.graph.Lam) ** 2

        J = G2 / (self.gamma + G2)

        return self.graph.scale_spectral(self.signal.Y, J)



from graph.graphs import Graph, ProductGraph
from graph.signals import PartiallyObservedGraphSignal, PartiallyObservedProductGraphSignal
from graph.filters import UnivariateFilterFunction, MultivariateFilterFunction
from typing import Union
from numpy import eye as I, diag
from numpy.linalg import solve, inv


class GraphSignalReconstructor:

    def __init__(self, graph: Graph, signal: PartiallyObservedGraphSignal, filter: UnivariateFilterFunction, gamma: float):
        self.graph = graph
        self.signal = signal
        self.filter = filter
        self.gamma = gamma
        self.graph.decompose()

    def get_f(self):
        H2 = self.graph.U @ diag(self.filter(self.graph.lam) ** 2) @ self.graph.U.T
        y_ = self.signal.down_project_signal(self.signal.y)
        M = self.signal.down_project_operator(H2) + self.gamma * I(self.signal.N_)
        return H2 @ self.signal.up_project_signal(solve(M, y_))

    def get_sig(self):
        H2 = self.graph.U @ diag(self.filter(self.graph.lam) ** 2) @ self.graph.U.T
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



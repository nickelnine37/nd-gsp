from graph.graphs import Graph, ProductGraph, BaseGraph
from graph.signals import GraphSignal, ProductGraphSignal
from graph.filters import GraphFilter, SpaceTimeGraphFilter
from typing import Union
from numpy import ndarray


def smooth_graph_signal(signal: ndarray, graph: BaseGraph, filter: GraphFilter, gamma: float) -> ndarray:
    """
    Smooth a graph signal according to the Bayesian model

    Parameters
    ----------
    signal
    graph
    filter
    gamma

    Returns
    -------

    """

    graph.decompose()
    g2 = filter(graph.lam) ** 2
    j = g2 / (gamma + g2)
    return graph.scale_spectral(signal, j)


class GraphSignalSmoother:

    def __init__(self, graph: Graph, signal: GraphSignal, filter: GraphFilter, gamma: float):

        self.graph = graph
        self.signal = signal
        self.filter = filter
        self.gamma = gamma

    def smooth(self):
        g2 = self.filter(self.graph.lam) ** 2
        j = g2 / (self.gamma + g2)
        return self.graph.scale_spectral(self.signal.y, j)


class ProductGraphSignalSmoother:

    def __init__(self, graph: ProductGraph, signal: ProductGraphSignal, filter: Union[GraphFilter, SpaceTimeGraphFilter], gamma: float):

        self.graph = graph
        self.signal = signal
        self.filter = filter
        self.gamma = gamma

    def smooth(self):

        if isinstance(self.filter, SpaceTimeGraphFilter):
            G2 = self.filter(self.graph.LamN, self.graph.LamT) ** 2
        else:
            G2 = self.filter(self.graph.Lam) ** 2

        J = G2 / (self.gamma + G2)

        return self.graph.scale_spectral(self.signal.Y, J)



from typing import Union

import networkx as nx
from numpy import ndarray

from graph.filters import _FilterFunction
from graph.graphs import BaseGraph
from models.reconstruction.univariate import UnivariateGraphSignalReconstructor
from models.reconstruction.multivariate import MultivariateGraphSignalReconstructor


def reconstruct_graph_signal(signal: ndarray,
                             graph: Union[BaseGraph, ndarray, nx.Graph],
                             filter_function: _FilterFunction,
                             gamma: float,
                             compute_logvar: bool=False):


    if signal.ndim == 1:
        reconstructor = UnivariateGraphSignalReconstructor(signal, graph, filter_function, gamma)

    elif signal.ndim > 1:
        reconstructor = MultivariateGraphSignalReconstructor(signal, graph, filter_function, gamma)
    else:
        raise ValueError()

    if compute_logvar:
        return reconstructor.compute_mean(), reconstructor.compute_logvar()

    else:
        return reconstructor.compute_mean()





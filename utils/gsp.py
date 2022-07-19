import numpy as np
from numpy import ndarray, diag
from scipy.sparse import csr_array, spmatrix, triu as sptriu, tril as sptril
from typing import Union
import networkx as nx
from graph.graphs import BaseGraph, Graph
from graph.filters import FilterFunction


def check_valid_adjacency(A: Union[ndarray, spmatrix], directed=False, self_loops=False) -> bool:
    """
    Perform some basic checks on the adjacency matrix. Returns true if no problems are
    found, else raises an assertion error

    Parameters
    ----------
    A               Adjacency matrix
    directed        Whether graph is directed
    self_loops      Whether self-loops are allowed

    """

    assert isinstance(A, Union[ndarray, spmatrix]), 'A must be an ndarray or an spmatrix'

    assert A.ndim == 2, 'A must be 2-dimensional'
    assert A.shape[0] == A.shape[1], 'A must be square'

    if not directed:
        assert (A.T != A).sum() == 0, 'A is not symmetric'

    if not self_loops:
        assert not A.diagonal().any(), 'A has non-zero entries along the diagonal'

    return True


def check_valid_laplacian(L: Union[ndarray, spmatrix]) -> bool:
    """
    Perform some basic checks on the Laplacian matrix. Returns true if no problems are
    found, else raises an assertion error. Directed graphs are not permitted here.

    Parameters
    ----------
    L               Laplacian matrix
    """

    assert isinstance(L, Union[ndarray, spmatrix]), 'L must be an ndarray or an spmatrix'

    assert L.ndim == 2, 'L must be 2-dimensional'
    assert L.shape[0] == L.shape[1], 'L must be square'

    if isinstance(L, ndarray):
        assert ((-np.triu(L, 1) - np.tril(L, -1)).sum(0) == L.diagonal()).all(), 'Laplacian matrix does not appear to have come from a valid adjacency matrix'

    if isinstance(L, spmatrix):
        assert (csr_array((-sptriu(L, 1) - sptril(L, -1))).sum(1) == L.diagonal()).all(), 'Laplacian matrix does not appear to have come from a valid adjacency matrix'

    return True


def check_valid_graph( graph: Union[BaseGraph, ndarray, nx.Graph]) -> BaseGraph:
    """
    The input can be a graph.BaseGraph, an adjacency matrix or a nx.Graph. This function
    checks whether the given input is valid, and coerces it into a BaseGraph.
    """

    if isinstance(graph, BaseGraph):
        return graph

    elif isinstance(graph, ndarray):
        return Graph.from_adjacency(graph)

    elif isinstance(graph, nx.Graph):
        return Graph.from_networkx(graph)

    else:
        raise TypeError('argument `graph` should be an ndarray, a Graph or a nx.Graph')


def check_compatible(signal: ndarray,
                     graph: BaseGraph,
                     filter_function: FilterFunction) -> None:
    """
    Make sure that the signal, graph and filter function are all mutually compaible. Perform the following checks:

        * Checks all the types are correct
        * The signal length is the same as the number of nodes in the graph, if 1D.
        * The signal has the correct tensor shape, if the graph is a product graph.
        * The filter function has the correct number of dimensions, if the graph is a product graph.

    returns None if all checks pass, raises an assertion error if there is a problem.
    """

    assert isinstance(signal, ndarray)
    assert isinstance(graph, BaseGraph)
    assert isinstance(filter_function, FilterFunction)

    assert signal.ndim == graph.ndim, f'The graph and the signal have a different number of dimenions ({graph.ndim} and {signal.ndim} respectively)'
    assert signal.shape == graph.signal_shape, f'The graph and signal have incompatible shapes: {graph.signal_shape} vs {signal.shape}'
    assert filter_function.ndim == graph.ndim, f'The filter function and the graph have a different number of dimenions ({filter_function.ndim} and {graph.ndim} respectively)'
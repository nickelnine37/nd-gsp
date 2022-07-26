import networkx as nx
from graph.graphs import BaseGraph, Graph
from graph.filters import _FilterFunction
from numpy import ndarray
from typing import Union


def check_valid_graph(graph: Union[BaseGraph, ndarray, nx.Graph]) -> BaseGraph:
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
        raise TypeError(f'argument `graph` should be an ndarray, a Graph or a nx.Graph but it is {type(graph)}')


def check_compatible(signal: ndarray = None,
                     graph: BaseGraph= None,
                     filter_function: _FilterFunction = None) -> None:
    """
    Make sure that the signal, graph and filter function are all mutually compaible. Perform the following checks:

        * Checks all the types are correct
        * The signal length is the same as the number of nodes in the graph, if 1D.
        * The signal has the correct tensor shape, if the graph is a product graph.
        * The filter function has the correct number of dimensions, if the graph is a product graph.

    All arguments are optional.

    returns None if all checks pass, raises an assertion error if there is a problem.
    """

    if signal is not None:
        assert isinstance(signal, ndarray)

    if graph is not None:
        assert isinstance(graph, BaseGraph)

    if filter_function is not None:
        assert isinstance(filter_function, _FilterFunction)

    if signal is not None and graph is not None:
        assert signal.ndim == graph.ndim, f'The graph and the signal have a different number of dimenions ({graph.ndim} and {signal.ndim} respectively)'
        assert signal.shape == graph.signal_shape, f'The graph and signal have incompatible shapes: {graph.signal_shape} vs {signal.shape}'

    # if filter_function is not None and graph is not None:
    #     assert filter_function.ndim == graph.ndim, f'The filter function and the graph have a different number of dimenions ({filter_function.ndim} and {graph.ndim} respectively)'
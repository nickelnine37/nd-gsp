import pytest
from pykronecker import KroneckerSum
import sys
import os
from ndgsp.graph.filters import UnivariateFilterFunction, FilterFunction, MultivariateFilterFunction
from ndgsp.graph.graphs import Graph, ProductGraph, BaseGraph
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import get_random_A, get_random_L, get_random_nx


def assert_gft(graph: BaseGraph):
    signal = np.random.normal(size=graph.signal_shape)
    assert np.allclose(signal, graph.rGFT(graph.GFT(signal)), rtol=1e-2, atol=1e-4)


def assert_filter(graph: BaseGraph, filter_func: FilterFunction):
    signal = np.random.normal(size=graph.signal_shape)
    graph.filter(signal, filter_func)


def test_graph():

    N = 100
    filter_func = UnivariateFilterFunction.diffusion(beta=1)

    for array_type in ['numpy', 'jax']:

        A = get_random_A(N, array_type)
        L = get_random_L(N, array_type)
        G = get_random_nx(N)

        for graph in [Graph.from_adjacency(A), Graph.from_laplacian(L), Graph.from_networkx(G)]:
            assert_gft(graph)
            assert_filter(graph, filter_func)

    for graph in [Graph.chain(N), Graph.loop(N), Graph.random_tree(N), Graph.random_connected(N), Graph.fully_connected(N)]:
        assert_gft(graph)
        assert_filter(graph, filter_func)

    with pytest.raises(ValueError):
        Graph()

    with pytest.raises(ValueError):
        Graph(A=A, L=L)

    with pytest.raises(ValueError):
        Graph(graph=A)

    with pytest.raises(ValueError):
        assert_filter(graph, graph)

    graph.__repr__()
    graph.__str__()


def test_product_graph():

    Ns = [10, 20, 30]
    filter_funcs = [UnivariateFilterFunction.diffusion(beta=1), MultivariateFilterFunction.diffusion(beta=[1] * len(Ns))]

    for array_type in ['numpy', 'jax']:

        As = []
        Ls = []
        Gs = []

        for N in Ns:

            As.append(get_random_A(N, array_type))
            Ls.append(get_random_L(N, array_type))
            Gs.append(get_random_nx(N))

        A = KroneckerSum(As)
        L = KroneckerSum(Ls)

        for graph in [ProductGraph.from_adjacency(A), ProductGraph.from_laplacian(L), ProductGraph.from_networkx(Gs)]:
            assert_gft(graph)

            for filter_func in filter_funcs:
                assert_filter(graph, filter_func)

    for graph in [ProductGraph.lattice(*Ns), ProductGraph.image(*Ns[:-1])]:

        assert_gft(graph)

        for filter_func in filter_funcs:
            assert_filter(graph, filter_func)

    with pytest.raises(ValueError):
        ProductGraph()

    with pytest.raises(ValueError):
        ProductGraph(A=A, L=L)

    with pytest.raises(ValueError):
        ProductGraph(graphs=[A, L])

    graph.__repr__()
    graph.__str__()

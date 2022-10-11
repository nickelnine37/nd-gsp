from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from numpy.linalg import eigh
import networkx as nx
from typing import Union, List

from pykronecker.base import KroneckerOperator
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from pykronecker import KroneckerSum, KroneckerProduct
# from src.utils.linalg import spdiag
# from src.utils.gsp import check_valid_adjacency, check_valid_laplacian
from ndgsp.graph.filters import FilterFunction, UnivariateFilterFunction, MultivariateFilterFunction

from ndgsp.utils.types import Numeric, Array, Operator
import jax.numpy as jnp


class BaseGraph(ABC):
    """
    Base class defining some behaviour graph classes should implement
    """

    # inheriting classes need to fill these variables

    N: int = None                                    # number of nodes in the graph
    ndim: int = None                                 # 1 for SimpleGraph, n for product graph
    signal_shape: tuple = None                       # the shape that signals defined on this graph should be

    A: Operator = None                               # dense adjacency matrix
    L: Operator = None                               # dense Laplacian
    graph: nx.Graph = None                           # networx graph

    decomposed: bool = False                         # whether eigendecomposition has been performed
    U: Operator = None                               # eigenvector matrix
    lam: ndarray = None                              # eigenvalues
    lams: ndarray = None                             # separated eigenvalues

    @abstractmethod
    def _decompose(self):
        """
        Perform eigendecomposition on the graph Laplacian. This never needs to be called by the end user.
        Instead, it is called automatically when the user attempts to access one of the attributes `U`,
        `lam` or `lams`. See __getattribute__ for details.
        """
        raise NotImplemented

    def GFT(self, Y: ndarray) -> ndarray:
        """
        Perform Graph Fourier Transform on a graph signal Y
        """
        return self.U.T @ Y

    def rGFT(self, Z: ndarray) -> ndarray:
        """
        Perform reverse Graph Fourier Transform on a fourier coefficients Z
        """
        return self.U @ Z

    def scale_spectral(self, Y: ndarray, G: ndarray) -> ndarray:
        """
        Scale the graph fourier coefficients of a signal Y by the function G.
        """
        return self.rGFT(G * self.GFT(Y))

    def get_G(self, filter_func: FilterFunction):

        if isinstance(filter_func, UnivariateFilterFunction):
            return filter_func(self.lam)

        elif isinstance(filter_func, MultivariateFilterFunction):
            return filter_func(self.lams)

        else:
            raise ValueError(f'filter_func should be an instance of graph.filters.FilterFunction, but it is {type(filter_func)}')

    def filter(self, Y: ndarray, filter_func: FilterFunction) -> ndarray:
        """
        Filter a graph signal Y given a graph filter function filter_func
        """

        return self.scale_spectral(Y, self.get_G(filter_func))

    def __getattribute__(self, item):
        """
        This is a small magic hack. It means that whenever we attempt to access one of the
        attributes `U`, `lam` or `lams`, we intercept and call self.decompose() first. This
        means we never have to call decompose() manually, and only call it lazily when needed.
        """

        if item in ['U', 'lam', 'lams']:
            self._decompose()

        return super().__getattribute__(item)

    @staticmethod
    def check_valid_adjacency(A: Array, directed=False, self_loops=False) -> bool:
        """
        Perform some basic checks on the adjacency matrix. Returns true if no problems are
        found, else raises an assertion error

        Parameters
        ----------
        A               Adjacency matrix
        directed        Whether graph is directed
        self_loops      Whether self-loops are allowed

        """

        assert isinstance(A, Array), f'A must be an array type, but it is {type(A)}'

        assert A.ndim == 2, 'A must be 2-dimensional'
        assert A.shape[0] == A.shape[1], 'A must be square'

        if not directed:
            assert (A.T != A).sum() == 0, f'A is not symmetricL (A.T != A).sum() = {(A.T != A).sum()}'

        if not self_loops:
            assert not A.diagonal().any(), 'A has non-zero entries along the diagonal'

        return True

    @staticmethod
    def check_valid_laplacian(L: Array) -> bool:
        """
        Perform some basic checks on the Laplacian matrix. Returns true if no problems are
        found, else raises an assertion error. Directed graphs are not permitted here. Self
        loops are not detectable.

        Parameters
        ----------
        L               Laplacian matrix
        """

        assert isinstance(L, Array), f'L must be an array type, but it is {type(L)}'

        assert L.ndim == 2, 'L must be 2-dimensional'
        assert L.shape[0] == L.shape[1], 'L must be square'
        assert ((-np.triu(L, 1) - np.tril(L, -1)).sum(0) == L.diagonal()).all(), 'Laplacian matrix does not appear to have come from a valid adjacency matrix'

        return True

    @staticmethod
    def A_to_L(A: Array):
        return np.diag(A.sum(0)) - A

    @staticmethod
    def L_to_A(L: Array):
        return np.diag(L.diagonal()) - L


class Graph(BaseGraph):

    @classmethod
    def from_networkx(cls, graph: nx.Graph):
        """
        Instantiate class from networkx graph
        """
        return cls(graph=graph)

    @classmethod
    def from_adjacency(cls, A: Array):
        """
        Instantiate class from adjacency matrix (ndarray or spamtrix)
        """
        return cls(A=A)

    @classmethod
    def from_laplacian(cls, L: Array):
        """
        Instantiate class from Laplacian matrix (ndarray or spamtrix)
        """
        return cls(L=L)

    @classmethod
    def chain(cls, N: int):
        """
        Create a chain graph of length N
        """
        A = np.zeros((N, N))
        A[range(0, N - 1), range(1, N)] = 1
        A = A + A.T
        return cls(A=A)

    @classmethod
    def loop(cls, N: int):
        """
        Create a circular loop graph of length N
        """
        A = np.zeros((N, N))
        A[range(0, N - 1), range(1, N)] = 1
        A[0, -1] = 1
        A = A + A.T
        return cls(A=A)

    @classmethod
    def random_tree(cls, N: int, seed: int=0):
        """
        Create a random tree graph of size N
        """
        np.random.seed(seed)

        X = np.random.uniform(0, 1, size=(N, 2))
        D = squareform(pdist(X))
        A = minimum_spanning_tree(D).toarray().astype(bool)
        A = A + A.T

        return cls(A=A.astype(float))

    @classmethod
    def random_connected(cls, N: int, seed: int=0):
        """
        Create a random graph with edges given by the perturbed minimum spanning
        tree algorithm. The resulting graph is fully connected, but is not a tree.
        """
        np.random.seed(seed)

        X = np.random.uniform(0, 1, size=(N, 2))
        D = squareform(pdist(X))
        A = minimum_spanning_tree(D).toarray().astype(bool)
        A = A + A.T

        X_ = X + 0.1 * np.random.uniform(0, 1, size=(N, 2))
        D_ = squareform(pdist(X_))
        A_ = minimum_spanning_tree(D_).toarray().astype(bool)
        A += A_ + A_.T

        return cls(A=A.astype(float))

    @classmethod
    def fully_connected(cls, N: int):
        """
        Create a fully connected graph with no self-loops
        """

        A = np.ones((N, N))
        A[range(N), range(N)] = 0
        return cls(A=A)

    def __init__(self,
                 A: Array = None,
                 L: Array = None,
                 graph: nx.Graph = None):
        """
        Create an undirected graph with no selp-loops. Can be instantiated by providing one of:

        graph       A networkx graph
        A           An adjacency matrix
        L           A Laplacian matrix

        However, this class in instantiated, the following instance attributes are created in the process:

        self.graph      A networkx graph representation
        self.A          A dense ndarray adjaceny matrix
        self.L          A dense ndarray Laplacian matrix

        """

        # check at least one argument is provided
        if all([graph is None, A is None, L is None]):
            raise ValueError('One of the arguments graph, A, or L must be provided')

        # check no more than one argument is provided
        if sum([graph is None, A is None, L is None]) < 2:
            raise ValueError('Only one of the arguments graph, A, or L should be provided')

        # handle nx.Graph argument
        if graph is not None:

            if not isinstance(graph, nx.Graph):
                raise ValueError('graph must be an instance of networkx.Graph')

            self.graph = graph
            self.A = nx.to_scipy_sparse_array(self.graph).toarray()
            self.L = np.diag(self.A.sum(0)) - self.A

        # handle A argument
        elif A is not None:

            self.check_valid_adjacency(A, directed=False, self_loops=False)
            self.A = A
            self.L = self.A_to_L(A)
            self.graph = nx.from_numpy_array(self.A)

        # handle L argument
        elif L is not None:

            self.check_valid_laplacian(L)
            self.L = L
            self.A = self.L_to_A(L)
            self.graph = nx.from_numpy_array(self.A)

        self.N = len(self.A)
        self.signal_shape = (self.N, )
        self.ndim = 1

    def _decompose(self):
        """
        Perform eigendecomposition of the Laplacian and update internal state accordingly.
        """
        if not self.decomposed:
            self.lam, self.U = eigh(self.L)
            self.decomposed = True

    def __repr__(self):
        return f'Graph(N={self.N})'

    def __str__(self):
        return f'Graph(N={self.N})'


class ProductGraph(BaseGraph):

    def __init__(self,
                 A: KroneckerSum = None,
                 L: KroneckerSum = None,
                 graphs: List[Graph] = None):
        """
        Initialise a Cartesian product graph from a sequence of graphs

        Parameters
        ----------
        *graphs     a sequence of graph objects
        """

        # check at least one argument is provided
        if all([graphs is None, A is None, L is None]):
            raise ValueError('One of the arguments graphs, A, or L must be provided')

        # check no more than one argument is provided
        if sum([graphs is None, A is None, L is None]) < 2:
            raise ValueError('Only one of the arguments graphs, A, or L should be provided')

        if graphs is not None:

            assert len(graphs) > 1, 'At least two graphs should be passed'

            if not all(isinstance(graph, Graph) for graph in graphs):
                raise ValueError('All arguments should be Graph objects')

            self.graphs = graphs
            self.A = KroneckerSum([graph.A for graph in graphs])
            self.L = KroneckerSum([graph.L for graph in graphs])

        elif A is not None:

            assert isinstance(A, KroneckerSum), f'A should be a KroneckerSum but it is a {type(A)}'
            self.graphs = [Graph.from_adjacency(A_) for A_ in A.As]
            self.A = A
            self.L = KroneckerSum([graph.L for graph in self.graphs])

        elif L is not None:

            assert isinstance(L, KroneckerSum), f'L should be a KroneckerSum but it is a {type(L)}'
            self.graphs = [Graph.from_laplacian(L_) for L_ in L.As]
            self.A = KroneckerSum([graph.A for graph in self.graphs])
            self.L = L

        self.ndim = len(self.graphs)
        self.N = int(np.prod([graph.N for graph in self.graphs]))
        self.signal_shape = tuple(graph.N for graph in self.graphs)

    @classmethod
    def from_networkx(cls, graphs: list[nx.Graph]):
        return cls(graphs=[Graph.from_networkx(graph) for graph in graphs])

    @classmethod
    def from_adjacency(cls, A: KroneckerSum):
        """
        Instantiate class from adjacency matrix (ndarray or spamtrix)
        """
        return cls(A=A)

    @classmethod
    def from_laplacian(cls, L: KroneckerSum):
        """
        Instantiate class from Laplacian matrix (ndarray or spamtrix)
        """
        return cls(L=L)

    @classmethod
    def lattice(cls, *Ns):
        """
        Create a lattice graph with Ni ticks in each dimension.
        """
        graphs = [Graph.chain(N) for N in Ns]
        return cls(graphs=graphs)

    @classmethod
    def image(cls, width: int, height: int):
        """
        Create a graph to represent a colour image, given an example image `like`.
        """
        graphs = [Graph.chain(width), Graph.chain(height), Graph.fully_connected(3)]
        return cls(graphs=graphs)

    def _decompose(self):
        """
        Perform eigendecomposition of the Laplacian and update internal state accordingly.
        """

        if not self.decomposed:

            for graph in self.graphs:
                graph._decompose()

            self.U = KroneckerProduct([graph.U for graph in self.graphs])

            # DO NOT ASSIGN lams DIRECTLY TO INSTANCE
            lams = np.array(np.meshgrid(*[g.lam for g in self.graphs], indexing='ij'))

            # BECAUSE OTHERWISE THIS WOULD CASUE __getattribute__ INFINITE RECURSION
            self.lam = lams.sum(0)

            # ASSIGN IT HERE
            self.lams = lams

            self.decomposed = True

    def __repr__(self):
        return f'Graph(N={" x ".join([str(graph.N) for graph in self.graphs])})'

    def __str__(self):
        return f'Graph(N={" x ".join([str(graph.N) for graph in self.graphs])})'




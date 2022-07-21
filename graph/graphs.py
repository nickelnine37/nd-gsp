import numpy as np
from numpy import ndarray
from numpy.linalg import eigh
import networkx as nx
from typing import Union
from scipy.sparse import spmatrix, csr_array
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from utils.kronecker import KroneckerSum, KroneckerProduct
from utils.linalg import spdiag
from utils.gsp import check_valid_adjacency, check_valid_laplacian


class BaseGraph:
    """
    Base class defining some behaviour graph classes should implement
    """

    # inheriting classes need to fill these variables

    N: int = None                                    # number of nodes in the graph
    ndim: int = None                                 # 1 for SimpleGraph, n for product graph
    signal_shape: tuple = None                       # the shape that signals defined on this graph should be

    A: Union[ndarray, KroneckerSum] = None           # dense adjacency matrix
    A_: Union[spmatrix, KroneckerSum] = None         # sparse adjacancy matrix
    L: Union[ndarray, KroneckerSum] = None           # dense Laplacian
    L_: Union[spmatrix, KroneckerSum] = None         # sparse Laplacian
    graph: nx.Graph = None                           # networx graph

    decomposed: bool = False                         # whether eigendecomposition has been performed
    U: Union[ndarray, KroneckerProduct] = None       # eigenvector matrix
    lam: ndarray = None                              # eigenvalue vector

    def _decompose(self):
        """
        Perform eigendecomposition on the graph Laplacian. This never needs to be called by the end user.
        Instead, it is called automatically when the user attempts to access one of the attributes `U`,
        `lam` or `lams`. See __getattribute__ for details.
        """
        return NotImplemented

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

    def scale_spectral(self, Y: ndarray, G: ndarray):
        """
        Scale the graph fourier coefficients of a signal Y by the function G.
        """
        return self.rGFT(G * self.GFT(Y))

    def __getattribute__(self, item):
        """
        This is a small magic hack. It means that whenever we attempt to access one of the
        attributes `U`, `lam` or `lams`, we intercept and call self.decompose() first. This
        means we never have to call decompose() manually, and only call it lazily when needed.
        """

        if item in ['U', 'lam', 'lams']:
            self._decompose()

        return super().__getattribute__(item)


class Graph(BaseGraph):

    @classmethod
    def from_networkx(cls, graph: nx.Graph):
        """
        Instantiate class from networkx graph
        """
        return cls(graph=graph)

    @classmethod
    def from_adjacency(cls, A: Union[ndarray, spmatrix]):
        """
        Instantiate class from adjacency matrix (ndarray or spamtrix)
        """
        return cls(A=A)

    @classmethod
    def from_laplacian(cls, L: Union[ndarray, spmatrix]):
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

        x = np.random.randn(N, 1)
        D = squareform(pdist(x))
        A = minimum_spanning_tree(D).toarray().astype(bool)
        A = A + A.T

        return cls(A=A.astype(float))

    @classmethod
    def random_connected(cls, N: int, n_loops: int=2, seed: int=0):
        """
        Create a random graph with edges that is the union of `n_loops` runs of the
        perturbed minimum sopanning tree algorithm. The resulting graph is fully
        connected, but is not a tree.
        """
        np.random.seed(seed)

        x = np.random.randn(N, 1)
        D = squareform(pdist(x))
        A = minimum_spanning_tree(D).toarray().astype(bool)
        A = A + A.T

        for i in range(n_loops - 1):
            D = squareform(pdist(x + np.random.randn(N, 1) / 50))
            A_ = minimum_spanning_tree(D).toarray().astype(bool)
            A += (A_ + A_.T)

        return cls(A=A.astype(float))


    def __init__(self,
                 graph: nx.Graph = None,
                 A: Union[ndarray, spmatrix] = None,
                 L: Union[ndarray, spmatrix] = None):
        """
        Create an undirected graph with no selp-loops. Can be instantiated by providing one of:

        graph       A networkx graph
        A           An adjacency matrix
        L           A Laplacian matrix

        However this class in instantiated, the following instance attributes are created in the process:

        self.graph      A networkx graph representation
        self.A          A dense ndarray adjaceny matrix
        self.A_         A sparse csr_array adjacency matrix
        self.L          A dense ndarray Laplacian matrix
        self.L_         A sparse csr_array Laplacian matrix

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
            self.A_ = csr_array(nx.adjacency_matrix(self.graph))
            self.A = self.A_.toarray()
            self.L_ = csr_array(nx.laplacian_matrix(self.graph))
            self.L = self.L_.toarray()

        # handle A argument
        elif A is not None:

            if not isinstance(A, Union[ndarray, spmatrix]):
                raise ValueError(f'A should be an numpy array or a scipy sparse matrix, but it is {type(A)}')

            check_valid_adjacency(A, directed=False, self_loops=False)

            self.A_ = csr_array(A)

            if isinstance(A, ndarray):
                self.A = A

            elif isinstance(A, spmatrix):
                self.A = self.A_.toarray()

            self.L_ = spdiag(self.A_.sum(1)) - self.A_
            self.L = self.L_.toarray()
            self.graph = nx.from_scipy_sparse_array(self.A_)

        # handle L argument
        elif L is not None:

            if not isinstance(L, Union[ndarray, spmatrix]):
                raise ValueError(f'L should be an numpy array or a scipy sparse matrix, but it is {type(L)}')

            check_valid_laplacian(L)

            self.L_ = csr_array(L)

            if isinstance(L, ndarray):
                self.L = L

            elif isinstance(L, spmatrix):
                self.L = self.L_.toarray()

            self.A_ = -self.L_ + spdiag(self.L_.diagonal())
            self.A = self.A_.toarray()
            self.graph = nx.from_scipy_sparse_array(self.A_)

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

    def __init__(self, *graphs):
        """
        Initialise a Cartesian product graph from a sequence of graphs

        Parameters
        ----------
        *graphs     a sequence of graph objects
        """

        assert len(graphs) > 1, 'At least two graphs should be passed'
        assert all(isinstance(graph, Graph) for graph in graphs), 'All arguments should be Graph objects'

        self.graphs = graphs
        self.ndim = len(graphs)
        self.N = int(np.prod([graph.N for graph in graphs]))
        self.signal_shape = tuple(graph.N for graph in reversed(graphs))
        self.lams = None

        self.A = KroneckerSum(*[graph.A for graph in graphs])
        self.A_ = KroneckerSum(*[graph.A_ for graph in graphs])
        self.L = KroneckerSum(*[graph.L for graph in graphs])
        self.L_ = KroneckerSum(*[graph.L_ for graph in graphs])

    @classmethod
    def lattice(cls, *Ns):
        """
        Create a lattice graph with Ni ticks in each dimension.
        """
        graphs = [Graph.chain(N) for N in Ns]
        return cls(*graphs)

    def _decompose(self):
        """
        Perform eigendecomposition of the Laplacian and update internal state accordingly.
        """

        if not self.decomposed:

            for graph in self.graphs:
                graph._decompose()

            self.U = KroneckerProduct(*[graph.U for graph in self.graphs])

            # DO NOT ASSIGN lams DIRECTLY TO INSTANCE
            lams = np.array(np.meshgrid(*[g.lam for g in reversed(self.graphs)], indexing='ij'))

            # BECAUSE OTHERWISE THIS WOULD CASUE __getattribute__ INFINITE RECURSION
            self.lam = lams.sum(0)

            # ASSIGN IT HERE
            self.lams = lams

            self.decomposed = True

    def __repr__(self):
        return f'Graph(N={" x ".join([str(graph.N) for graph in self.graphs])})'

    def __str__(self):
        return f'Graph(N={" x ".join([str(graph.N) for graph in self.graphs])})'



def _run_tests():

    def test_graph():
        N = 100
        graph = Graph.random_tree(N)
        signal = np.random.randn(N)

        assert np.allclose(signal, graph.rGFT(graph.GFT(signal)))

    def test_product_graph():

        N1 = 100; N2 = 150
        graph = ProductGraph.lattice(N1, N2)
        signal = np.random.randn(N2, N1)

        assert np.allclose(signal, graph.rGFT(graph.GFT(signal)))

    test_graph()
    test_product_graph()

    print('all tests passed')


if __name__ == '__main__':

    _run_tests()
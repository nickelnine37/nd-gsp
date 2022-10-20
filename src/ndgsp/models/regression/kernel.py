from abc import ABC
from typing import Tuple, Iterable

from jax import jit
from pykronecker import KroneckerDiag, KroneckerIdentity, KroneckerProduct

from ndgsp.algorithms.cgm import solve_SPCGM, solve_CGM
from ndgsp.graph.filters import FilterFunction
from ndgsp.graph.graphs import BaseGraph
from ndgsp.models.model import Model, RealModel, LogisticModel
from ndgsp.utils.arrays import expand_dims, outer_product
from ndgsp.utils.types import Array, Signal
import numpy as np
from scipy.spatial.distance import pdist, squareform
import jax.numpy as jnp


class KernelModel(Model, ABC):

    def __init__(self,
                 X: Array,
                 Y: Signal,
                 graph: BaseGraph,
                 filter_func: FilterFunction,
                 gamma: float,
                 kernel_std: float):

        Y = np.ararray(Y)
        self.check_consistent(Y[0], graph)
        self.Y, self.S = self.get_Y_and_S(Y)

        assert X.shape[0] == self.Y.shape[0]

        self.X = jnp.asarray(X)
        self.graph = graph
        self.filter_func = filter_func
        self.gamma = gamma
        self.kernel_std = kernel_std

        self.K, self.lamK, self.V = self.get_kernel(self.X)
        self.U = KroneckerProduct([self.V] + [Ui for Ui in self.graph.U])

    def get_kernel(self, X: Array) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Return a square Gaussian kernel matrix such that K_{ij} = exp(-0.5 * (x[i] - x[j]) ** 2 / self.kernel_std ** 2)
        """
        D = jnp.asarray(squareform(pdist(X, metric='sqeuclidean')))
        K = jnp.exp(-0.5 * D / self.kernel_std ** 2)
        lamK, V = jnp.linalg.eigh(K)
        return K, lamK, V

    def set_kernel_std(self, kernel_std: float):
        self.kernel_std = kernel_std
        self.K, self.lamK, self.V = self.get_kernel(self.X)
        self.U = KroneckerProduct([self.V] + [Ui for Ui in self.graph.U])

    def get_G(self):
        return outer_product(self.lamK ** 0.5, self.graph.get_G(self.filter_func))


class KGR(RealModel, KernelModel):

    def __init__(self,
                 X: Array,
                 Y: Signal,
                 graph: BaseGraph,
                 filter_func: FilterFunction,
                 gamma: float,
                 kernel_std: float):

        super().__init__(X, Y, graph, filter_func, gamma, kernel_std)


class LogisticKGR(LogisticModel, KernelModel):

    def __init__(self,
                 X: Array,
                 Y: Signal,
                 graph: BaseGraph,
                 filter_func: FilterFunction,
                 gamma: float,
                 kernel_std: float):
        """
        Initialise a Logistic Graph Signal Reconstruction model. Pass in a binary-valued graph signal and
        a product graph. Missing values in the graph signal should be indicated with nans.
        """

        super().__init__(X, Y, graph, filter_func, gamma, kernel_std)

from abc import ABC
from typing import Iterable

import numpy as np
from jax import jit
from pykronecker import KroneckerDiag, KroneckerIdentity

from ndgsp.algorithms.cgm import solve_SPCGM, solve_CGM
from ndgsp.graph.filters import FilterFunction, MultivariateFilterFunction
from ndgsp.graph.graphs import ProductGraph
from ndgsp.models.model import Model, RealModel, LogisticModel
from ndgsp.utils.types import Signal, Array
import jax.numpy as jnp


class ReconstructionModel(Model, ABC):

    def get_G(self):
        return self.graph.get_G(self.filter_func)

    def __init__(self, signal: Signal, graph: ProductGraph, filter_func: FilterFunction, gamma: float):
        """
        Initialise a Graph Signal Reconstruction model. Pass in a real-valued graph signal and
        a product graph. Missing values in the graph signal should be indicated with nans.
        """

        self.check_consistent(signal, graph)
        self.Y, self.S = self.get_Y_and_S(signal)
        self.graph = graph
        self.gamma = gamma
        self.filter_func = filter_func
        self.U = self.graph.U


class GSR(RealModel, ReconstructionModel):
    """
    Graph Signal Reconstruction on a Cartesian product graph
    """

    def __init__(self, signal: Signal, graph: ProductGraph, filter_func: FilterFunction, gamma: float):
        """
        Initialise a Graph Signal Reconstruction model. Pass in a real-valued graph signal and
        a product graph. Missing values in the graph signal should be indicated with nans.
        """

        super().__init__(signal, graph, filter_func, gamma)


class LogisticGSR(LogisticModel, ReconstructionModel):

    def __init__(self, signal: Signal, graph: ProductGraph, filter_func: FilterFunction, gamma: float):
        """
        Initialise a Logistic Graph Signal Reconstruction model. Pass in a binary-valued graph signal and
        a product graph. Missing values in the graph signal should be indicated with nans.
        """

        super().__init__(signal, graph, filter_func, gamma)




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np

    test_lgsr = False
    test_gsr = False
    test_gsr_sample = True

    if test_lgsr:

        N = 100

        Y = np.random.randint(0, 2, size=(N, N)).astype(float)
        S = np.random.randint(0, 2, size=(N, N)).astype(bool)
        Y[S] = np.nan

        graph = ProductGraph.lattice(N, N)
        filter_func = MultivariateFilterFunction.diffusion([1, 1])

        model = LogisticGSR(Y, graph, filter_func, 1)

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(Y, interpolation='nearest')
        Y_ = model.compute_mean()
        print(Y_)
        ax[1].imshow(Y_ > 0.5)

        plt.show()

    if test_gsr:

        N = 100

        Y = np.random.normal(size=(N, N))
        S = np.random.randint(0, 2, size=(N, N)).astype(bool)
        Y[S] = np.nan

        graph = ProductGraph.lattice(N, N)
        filter_func = MultivariateFilterFunction.diffusion([1, 1])

        model = GSR(Y, graph, filter_func, 1)

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(Y, interpolation='nearest')
        ax[1].imshow(model.compute_mean())

        plt.show()

    if test_gsr_sample:

        from PIL import Image

        Y = np.asarray(Image.open('../../../assets/image1.jpg').convert('L')).astype(float)

        Y -= Y.mean()
        Y /= Y.std()

        print(Y)

        graph = ProductGraph.lattice(*Y.shape)
        filter_func = MultivariateFilterFunction.diffusion([1, 1])

        S = np.random.randint(0, 2, size=Y.shape).astype(bool)
        # Y = Y.at[S].set(np.nan)
        Y[S] = np.nan

        model = GSR(Y, graph, filter_func, 1)

        samples = model.sample(4)

        fig, axes = plt.subplots(ncols=2, nrows=2)

        for i, ax in enumerate(axes.ravel()):

            ax.imshow(samples[i])

        plt.show()


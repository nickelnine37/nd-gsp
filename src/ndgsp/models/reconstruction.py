from typing import Iterable

import numpy as np
from jax import jit
from pykronecker import KroneckerDiag, KroneckerIdentity

from ndgsp.algorithms.cgm import solve_SPCGM, solve_CMG
from ndgsp.graph.filters import FilterFunction, MultivariateFilterFunction
from ndgsp.graph.graphs import ProductGraph
from ndgsp.models.model import Model
from ndgsp.utils.types import Signal, Array
import jax.numpy as jnp


class GSR(Model):
    """
    Graph Signal Reconstruction on a Cartesian product graph
    """

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

    def set_gamma(self, gamma: float):
        self.gamma = gamma

    def set_beta(self, beta: float | Iterable):
        self.filter_func.set_beta(beta)

    def compute_mean(self, tol: float=1e-8, verbose: bool=False):

        DG = KroneckerDiag(self.graph.get_G(self.filter_func))
        DS = KroneckerDiag(self.S)

        A = DG @ self.graph.U.T @ DS @ self.graph.U @ DG + self.gamma * KroneckerDiag(jnp.ones_like(self.S))
        Phi = self.graph.U @ DG
        PhiT = Phi.T

        return solve_SPCGM(A_precon=A,
                           y=self.Y,
                           Phi=Phi,
                           PhiT=PhiT,
                           reltol=tol,
                           verbose=verbose)

    def sample(self, n_samples: int=1):

        DG = KroneckerDiag(self.graph.get_G(self.filter_func))
        DS = KroneckerDiag(self.S)
        Y_ = DG @ self.graph.U.T @ self.Y
        Q = DG @ self.graph.U.T @ DS @ self.graph.U @ DG + self.gamma * KroneckerDiag(jnp.ones_like(self.S))

        samples = []

        for _ in range(n_samples):

            Z1_ = DG @ self.graph.U.T @ DS @ np.random.normal(size=self.Y.shape)
            Z2_ = self.gamma ** 0.5 * np.random.normal(size=self.Y.shape)
            z = solve_CMG(Q, Z1_ + Z2_ + Y_)
            samples.append(self.graph.U @ DG @ z)

        if n_samples == 1:
            return samples[0]

        else:
            return samples


class LogisticGSR(Model):

    def __init__(self, signal: Signal, graph: ProductGraph, filter_func: FilterFunction, gamma: float):
        """
        Initialise a Logistic Graph Signal Reconstruction model. Pass in a binary-valued graph signal and
        a product graph. Missing values in the graph signal should be indicated with nans.
        """

        self.check_consistent(signal, graph)
        self.Y, self.S = self.get_Y_and_S(signal)
        self.graph = graph
        self.gamma = gamma
        self.filter_func = filter_func

    @staticmethod
    @jit
    def get_mu(alpha: Array):
        return (1 + jnp.exp(-alpha)) ** -1

    def _compute_alpha_star(self):

        DG = KroneckerDiag(self.graph.get_G(self.filter_func))
        DS = KroneckerDiag(self.S)
        I = KroneckerIdentity(like=self.graph.U)

        alpha_ = np.zeros_like(self.Y)
        da = 1

        while da > 1e-5:
            mu = self.get_mu(self.graph.U @ DG @ alpha_)
            Dmu_ = KroneckerDiag(mu * (1 - mu))
            H = DG @ self.graph.U.T @ DS @ Dmu_ @ DS @ self.graph.U @ DG + self.gamma * I
            x = DG @ self.graph.U.T @ DS @ (Dmu_ @ DS @ self.graph.U @ DG @ alpha_ + self.Y - mu)
            alpha_new = solve_CMG(H, x)
            da = np.abs(alpha_ - alpha_new).sum() / np.prod(alpha_.shape)
            alpha_ = alpha_new

        return alpha_

    def compute_mean(self):
        DG = KroneckerDiag(self.graph.get_G(self.filter_func))
        return self.get_mu(self.graph.U @ DG @ self._compute_alpha_star())

    def sample(self, n_samples: int=1):

        DG = KroneckerDiag(self.graph.get_G(self.filter_func))
        DS = KroneckerDiag(self.S)
        alpha_star = self._compute_alpha_star()

        mu = self.get_mu(self.graph.U @ DG @ alpha_star)
        Dmu_ = KroneckerDiag(mu * (1 - mu))
        Dmu_chol = Dmu_ ** 0.5

        H = DG @ self.graph.U.T @ DS @ Dmu_ @ DS @ self.graph.U @ DG + self.gamma * KroneckerIdentity(like=self.graph.U)

        samples = []

        for _ in range(n_samples):

            Z1_ = DG @ self.graph.U.T @ DS @ Dmu_chol @ np.random.normal(size=self.Y.shape)
            Z2_ = self.gamma ** 0.5 * np.random.normal(size=self.Y.shape)
            alpha_sample = alpha_star + solve_CMG(H, Z1_ + Z2_)
            samples.append(self.get_mu(self.graph.U @ DG @ alpha_sample))

        if n_samples == 1:
            return samples[0]

        else:
            return samples


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


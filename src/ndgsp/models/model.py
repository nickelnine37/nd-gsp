from abc import ABC, abstractmethod
from typing import Tuple, Iterable

from jax import jit
from pykronecker import KroneckerDiag, KroneckerIdentity

from ndgsp.algorithms.cgm import solve_SPCGM, solve_CGM
from ndgsp.graph.filters import FilterFunction
from ndgsp.graph.graphs import BaseGraph, ProductGraph
from ndgsp.utils.types import Signal, Array, Operator
import jax.numpy as jnp
import pandas as pd
import numpy as np


class Model(ABC):

    Y: Array
    S: Array
    graph: BaseGraph
    filter_func: FilterFunction
    U: Operator
    gamma: float

    @abstractmethod
    def get_G(self):
        raise NotImplementedError

    @abstractmethod
    def compute_mean(self, tol: float=1e-8, verbose: bool=False):
        raise NotImplementedError

    @abstractmethod
    def sample(self, n_samples: int=1):
        raise NotImplementedError

    def set_gamma(self, gamma: float):
        self.gamma = gamma

    def set_beta(self, beta: float | Iterable):
        self.filter_func.set_beta(beta)

    @staticmethod
    def get_Y_and_S(signal: Signal) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Convert a graph signal containing nans into two separate arrays. The first is a copy of signal, with nans
        replaced by 0. The second is a binary array containing zeros where nans were, and ones elsewhere
        """

        assert isinstance(signal, Signal), f'signal should be either a numpy array, jax array, or pandas DataFrame, but it is a {type(signal)}'
        assert signal.ndim > 1, f'signal should have more than 1 dimension, but it has {signal.ndim}'

        if isinstance(signal, pd.DataFrame):
            Y = jnp.asarray(signal.values.astype(float))

        elif isinstance(signal, np.ndarray):
            Y = jnp.asarray(signal.astype(float))

        else:
            Y = signal.astype(float)

        S_ = jnp.isnan(Y)

        return Y.at[S_].set(0), (~S_).astype(float)

    @staticmethod
    def check_consistent(signal: Signal, graph: BaseGraph):
        assert signal.ndim == graph.ndim, f'The signal and graph should have the same number of dimensions, but they have {signal.ndim} and {graph.ndim} respectively'
        assert signal.shape == graph.A.tensor_shape, f'The signal and graph should have consistent shapes, but they have {signal.shape} and {graph.A.tensor_shape} respectively'


class RealModel(Model, ABC):

    def compute_mean(self, tol: float=1e-8, verbose: bool=False):

        DG = KroneckerDiag(self.get_G())
        DS = KroneckerDiag(self.S)
        I = KroneckerIdentity(like=DG)

        A = DG @ self.U.T @ DS @ self.U @ DG + self.gamma * I
        Phi = self.U @ DG

        return solve_SPCGM(A_precon=A,
                           y=self.Y,
                           Phi=Phi,
                           reltol=tol,
                           verbose=verbose)

    def sample(self, n_samples: int=1):

        DG = KroneckerDiag(self.get_G())
        DS = KroneckerDiag(self.S)

        Y_ = DG @ self.U.T @ self.Y
        A = DG @ self.U.T @ DS @ self.U @ DG + self.gamma * KroneckerIdentity(like=DG)

        samples = []

        for _ in range(n_samples):

            Z1_ = DG @ self.U.T @ DS @ np.random.normal(size=self.Y.shape)
            Z2_ = self.gamma ** 0.5 * np.random.normal(size=self.Y.shape)
            z = solve_CGM(A, Z1_ + Z2_ + Y_)
            samples.append(self.U @ DG @ z)

        if n_samples == 1:
            return samples[0]

        else:
            return samples


class LogisticModel(Model, ABC):

    @staticmethod
    @jit
    def get_mu(alpha: Array):
        return (1 + jnp.exp(-alpha)) ** -1

    def _compute_alpha_star(self):

        DG = KroneckerDiag(self.get_G())
        DS = KroneckerDiag(self.S)
        I = KroneckerIdentity(like=DG)

        alpha_ = np.zeros_like(self.Y)
        da = 1

        while da > 1e-5:
            mu = self.get_mu(self.U @ DG @ alpha_)
            Dmu_ = KroneckerDiag(mu * (1 - mu))
            H = DG @ self.U.T @ DS @ Dmu_ @ DS @ self.U @ DG + self.gamma * I
            x = DG @ self.U.T @ DS @ (Dmu_ @ DS @ self.U @ DG @ alpha_ + self.Y - mu)
            alpha_new = solve_CGM(H, x)
            da = np.abs(alpha_ - alpha_new).sum() / np.prod(alpha_.shape)
            alpha_ = alpha_new

        return alpha_

    def compute_mean(self, tol: float=1e-8, verbose: bool=False):
        return self.get_mu(self.U @ KroneckerDiag(self.get_G()) @ self._compute_alpha_star())

    def sample(self, n_samples: int=1):

        DG = KroneckerDiag(self.get_G())
        DS = KroneckerDiag(self.S)
        alpha_star = self._compute_alpha_star()

        mu = self.get_mu(self.U @ DG @ alpha_star)
        Dmu_ = KroneckerDiag(mu * (1 - mu))
        Dmu_chol = Dmu_ ** 0.5

        H = DG @ self.U.T @ DS @ Dmu_ @ DS @ self.U @ DG + self.gamma * KroneckerIdentity(like=self.graph.U)

        samples = []

        for _ in range(n_samples):

            Z1_ = DG @ self.U.T @ DS @ Dmu_chol @ np.random.normal(size=self.Y.shape)
            Z2_ = self.gamma ** 0.5 * np.random.normal(size=self.Y.shape)
            alpha_sample = alpha_star + solve_CGM(H, Z1_ + Z2_)
            samples.append(self.get_mu(self.U @ DG @ alpha_sample))

        if n_samples == 1:
            return samples[0]

        else:
            return samples

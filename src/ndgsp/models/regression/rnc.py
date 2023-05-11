from abc import ABC

import numpy as np
from pykronecker import KroneckerDiag, KroneckerIdentity, KroneckerBlock, KroneckerBlockDiag

from ndgsp.algorithms.cgm import solve_CGM, solve_SPCGM
from ndgsp.graph.filters import FilterFunction, MultivariateFilterFunction
from ndgsp.graph.graphs import BaseGraph, ProductGraph
from ndgsp.models.model import Model, LogisticModel, RealModel
from ndgsp.utils.types import Array, Signal
import jax.numpy as jnp
from jax import jit


class NCModel(Model, ABC):

    def __init__(self,
                 X: Array,
                 Y: Signal,
                 graph: BaseGraph,
                 filter_func: FilterFunction,
                 gamma: float,
                 lam: float):

        self.check_consistent(Y, graph)
        self.Y, self.S = self.get_Y_and_S(Y)
        self.graph = graph
        self.gamma = gamma
        self.lam = lam
        self.filter_func = filter_func
        self.U = self.graph.U

        assert X.ndim == 2
        assert X.shape[0] == np.prod(self.Y.shape)
        self.X = jnp.asarray(X)


class RNC(RealModel, NCModel):

    def compute_mean(self, tol: float = 1e-8, verbose: bool = False):

        DG = KroneckerDiag(self.get_G())
        DS = KroneckerDiag(self.S)
        Xq = self.X[self.S.astype(bool).ravel(), :]
        lamM, UM = jnp.linalg.eigh(Xq.T @ Xq)
        DM = jnp.diag((lamM + self.lam) ** -0.5)

        M11 = DG @ self.U.T @ DS @ self.U @ DG + self.gamma * KroneckerIdentity(like=DG)
        M12 = DG @ self.U.T @ DS @ self.X @ UM @ DM

        M = KroneckerBlock([[M11, M12], [M12.T, np.eye(self.X.shape[1])]])
        Y_ = np.concatenate([self.Y.ravel(), self.X.T @ self.Y.ravel()])

        Phi = KroneckerBlockDiag([self.U @ DG, UM @ DM])

        theta, nits = solve_SPCGM(A_precon=M, Y=Y_, Phi=Phi)
        alpha = theta[:self.graph.N]
        beta = theta[self.graph.N:]

        return (alpha + self.X @ beta).reshape(self.Y.shape)

    def sample(self, n_samples: int = 1):
        pass

    def __init__(self,
                 X: Array,
                 Y: Signal,
                 graph: BaseGraph,
                 filter_func: FilterFunction,
                 gamma: float,
                 lam: float):

        super().__init__(X, Y, graph, filter_func, gamma, lam)


class LogisticRNC(LogisticModel, NCModel):

    def __init__(self,
                 X: Array,
                 Y: Signal,
                 graph: BaseGraph,
                 filter_func: FilterFunction,
                 gamma: float,
                 lam: float):

        super().__init__(X, Y, graph, filter_func, gamma, lam)

    @jit
    def get_mu(self, alpha: Array, beta: Array):
        return ((1 + jnp.exp(-alpha + self.X @ beta)) ** -1).reshape(self.graph.U.tensor_shape)

    def _compute_theta_star(self):

        DG = KroneckerDiag(self.get_G())
        DS = KroneckerDiag(self.S)
        I = KroneckerIdentity(like=DG)
        Xs = DS @ self.X

        alpha_ = np.zeros(self.graph.N)
        beta = np.zeros(self.X.shape[1])
        da = 1
        db = 1

        while (da + db) > 1e-5:

            mu = self.get_mu(self.U @ DG @ alpha_, beta)
            Dmu_ = KroneckerDiag(mu * (1 - mu))

            H11 = DG @ self.U.T @ DS @ Dmu_ @ DS @ self.U @ DG + self.gamma * I
            H12 = DG @ self.U.T @ DS @ Dmu_ @ Xs
            H22 = Xs.T @ Dmu_ @ Xs + self.lam * np.eye(self.X.shape[1])
            H = KroneckerBlock([[H11, H12], [H12.T, H22]])

            xx = (Dmu_ @ DS @ (self.U @ DG @ alpha_ + self.X @ beta) + (self.Y - mu).ravel())
            x1 = DG @ self.U.T @ DS @ xx
            x2 = Xs.T @ xx

            theta_new = solve_CGM(H, np.concatenate([x1, x2]))
            da = np.abs(alpha_ - theta_new[:self.graph.N]).sum() / self.graph.N
            db = np.abs(beta - theta_new[self.graph.N:]).sum() / self.X.shape[1]
            alpha_ = theta_new[:self.graph.N]
            beta = theta_new[self.graph.N:]

        return alpha_, beta

    def compute_mean(self, tol: float=1e-8, verbose: bool=False):
        alpha_, beta = self._compute_theta_star()
        return self.get_mu(self.U @ KroneckerDiag(self.get_G()) @ alpha_, beta)

    def sample(self, n_samples: int=1):

        DG = KroneckerDiag(self.get_G())
        DS = KroneckerDiag(self.S)
        I = KroneckerIdentity(like=DG)
        Xs = DS @ self.X

        alpha_, beta = self._compute_theta_star()
        theta_star = np.concatenate([alpha_, beta])

        mu = self.get_mu(self.U @ DG @ alpha_, beta)
        Dmu_ = KroneckerDiag(mu * (1 - mu))
        Dmu_chol = Dmu_ ** 0.5

        A = DG @ self.U.T @ DS @ Dmu_chol
        B = Xs.T @ Dmu_chol

        H = KroneckerBlock([[A @ A.T + self.gamma * I, A @ B.T],
                            [B @ A.T, B @ B.T + self.lam * np.eye(self.X.shape[1])]])

        samples = []

        for _ in range(n_samples):

            z1_ = np.random.normal(size=self.graph.N)
            z1_ = np.concatenate([A @ z1_, B @ z1_])
            z2_ = np.concatenate([self.gamma ** 0.5 * np.random.normal(size=self.graph.N), self.lam ** 0.5 * np.random.normal(size=self.X.shape[1])])
            theta_sample = theta_star + solve_CGM(H, z1_ + z2_)
            samples.append(self.get_mu(self.U @ DG @ theta_sample[:self.graph.N], theta_sample[self.graph.N:]))

        if n_samples == 1:
            return samples[0]

        else:
            return samples



if __name__ == '__main__':

    N1 = 4
    N2 = 5
    N3 = 6
    M = 3

    np.random.seed(0)

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    X = np.random.normal(size=(N1 * N2 * N3, M))
    Y = np.random.normal(size=(N1, N2, N3))
    Y[np.random.randint(2, size=(N1, N2, N3)).astype(bool)] = np.nan

    graph = ProductGraph.lattice(N1, N2, N3)
    ffunc = MultivariateFilterFunction.diffusion([0.2] * 3)

    model = RNC(X, Y, graph, ffunc, 1, 1)

    F = model.compute_mean()

    G = ffunc(graph.lams)

    Hi2 = graph.U @ KroneckerDiag(G ** -2) @ graph.U.T
    S = (~np.isnan(Y)).astype(int)
    DS = KroneckerDiag(S)
    Y_ = Y.copy()
    Y_[np.isnan(Y)] = 0

    P = np.block([[(DS + Hi2).to_array(), DS @ X], [X.T @ DS, X.T @ DS @ X + np.eye(M)]])
    theta = np.linalg.solve(P, np.block([Y_.ravel(), X.T @ Y_.ravel()]))
    alpha = theta[:graph.N]
    beta = theta[graph.N:]
    F_ = (alpha + X @ beta).reshape(Y.shape)

    print(F)

    print(F_)


    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(F.ravel())

    plt.plot(F_.ravel())

    plt.show()
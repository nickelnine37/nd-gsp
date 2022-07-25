import numpy as np
from numpy import ndarray, eye as I, diag
from typing import Union, Callable
import networkx as nx
from scipy.sparse import spmatrix

from algorithms.cgm import solve_SPCGM
from graph.filters import _FilterFunction, MultivariateFilterFunction, UnivariateFilterFunction
from graph.graphs import BaseGraph, ProductGraph
from models.reconstruction.reconstruction_utils import get_y_and_s
from utils.checks import check_compatible
from utils.linalg import vec, ten
from utils.kronecker import KroneckerBlock, KroneckerBlockDiag, KroneckerDiag

from numpy.linalg import eigh, solve
from scipy.optimize import minimize


class VarSolver:

    def __init__(self, Omega_Q: ndarray, Q: ndarray, X: ndarray, lam: float = 0.005):
        assert all([Omega_Q.shape == Q.shape]), f'Omega_Q and Q should all have the same shape, but they are {Omega_Q.shape} and {Q.shape} respectively'

        self.Omega_Q = Omega_Q
        self.X = X
        self.Q = Q.astype(bool)
        self.lam = lam
        self.params = None

    def rmse(self, Omega: ndarray):
        return (((Omega - self.predict()) ** 2).sum() / (np.prod(self.Q.shape))) ** 0.5

    def r_squared(self, Omega: ndarray):
        return 1 - ((Omega - self.predict()) ** 2).sum() / ((Omega - Omega.mean()) ** 2).sum()

    def _get_params(self):
        pass

    def predict(self):
        pass


class RRVarSolver(VarSolver):

    def __init__(self, Omega_Q: ndarray, Q: ndarray, X: ndarray, lam: float = 0.005):
        super().__init__(Omega_Q, Q, X, lam)

    def _get_params(self):
        Xq = self.X[vec(self.Q)]
        M = Xq.T @ Xq + self.lam * I(self.X.shape[1])
        y_ = self.X.T @ vec(self.Omega_Q)
        self.params = [solve(M, y_)]

    def predict(self):
        if self.params is None:
            self._get_params()
        return ten(self.X @ self.params[0], like=self.Omega_Q)


class RNCVarSolver(VarSolver):

    def __init__(self,
                 Omega_Q: ndarray,
                 Q: ndarray,
                 X: ndarray,
                 graph: BaseGraph,
                 filter_function: _FilterFunction,
                 gamma: float,
                 lam: float):

        super().__init__(Omega_Q, Q, X, lam)

        check_compatible(signal=Omega_Q, graph=graph, filter_function=filter_function)

        self.N = X.shape[0]
        self.Omega_Q = Omega_Q

        if isinstance(filter_function, MultivariateFilterFunction):
            G = filter_function(graph.lams)
        else:
            G = filter_function(graph.lam)

        Xq = X[vec(self.Q)]
        lamX, Psi = eigh(Xq.T @ Xq)
        DX = diag((lamX + lam) ** -0.5)
        P = X.shape[1]

        DQ = KroneckerDiag(self.Q)
        DG = KroneckerDiag(G)

        M11 = DG @ graph.U.T @ DQ @ graph.U @ DG + gamma * KroneckerDiag(np.ones_like(self.Q))
        M12 = DG @ graph.U.T @ DQ @ X @ Psi @ DX

        self.M = KroneckerBlock([[M11, M12],
                                 [M12.T, np.eye(P)]])

        self.Phi = KroneckerBlockDiag([graph.U @ DG, Psi @ DX])
        self.PhiT = self.Phi.T


    def _get_params(self, verbose=False, tol=1e-5):

        result, nits = solve_SPCGM(A_precon=self.M,
                                   y=np.concatenate([vec(self.Omega_Q), self.X.T @ vec(self.Omega_Q)]),
                                   Phi=self.Phi,
                                   PhiT=self.PhiT,
                                   reltol=tol,
                                   verbose=verbose)

        print(f'CGM completed in {nits} iterations')

        self.params = [ten(result[:self.N], like=self.Omega_Q), result[self.N:]]

    def predict(self, verbose=False):
        if self.params is None:
            self._get_params(verbose=verbose)
        return self.params[0] + ten(self.X @ self.params[1], like=self.Omega_Q)


class LFPVarSolver(VarSolver):

    def __init__(self,
                 Omega_Q: ndarray,
                 Q: ndarray,
                 X: ndarray,
                 graph: BaseGraph,
                 filter_function: _FilterFunction,
                 lam: float):

        super().__init__(Omega_Q, Q, X, lam)

        check_compatible(signal=Omega_Q, graph=graph, filter_function=filter_function)

        if isinstance(filter_function, UnivariateFilterFunction):
            self.x0 = np.array([0, 0, filter_function.beta, 0, filter_function.beta])
        elif isinstance(filter_function, MultivariateFilterFunction):
            self.x0 = np.array([0, 0] + filter_function.beta.tolist() + [0] + filter_function.beta.tolist())

        self.graph = graph
        self.filter_function = filter_function

        self.S_ = ten(self.X[:, 1], like=Q)
        self.A_ = ten(self.X[:, -2], like=Q)

    def H(self, Y: ndarray, beta: float | ndarray):
        """
        Apply the operation Y -> ten(H @ vec(Y)) efficiently for a filter defined by λ -> η(λ; β)
        """

        self.filter_function.set_beta(beta)

        if isinstance(self.filter_function, MultivariateFilterFunction):
            G = self.filter_function(self.graph.lams)
        else:
            G = self.filter_function(self.graph.lam)


        return self.graph.scale_spectral(Y, G)

    def Omega(self, v: ndarray) -> ndarray:
        """
        Return the estitenor for Omega for a given objective vector
        """

        if isinstance(self.filter_function, MultivariateFilterFunction):
            beta1 = v[2:2+self.filter_function.ndim]
            beta2 = v[3+self.filter_function.ndim:]

        else:
            beta1 = v[2]
            beta2 = v[4]

        return v[0] + v[1] * self.H(self.S_, beta1) + v[2+self.filter_function.ndim] * self.H(self.A_, beta2)

    def objective(self, v: ndarray):
        """
        The objective function to minimise
        """
        return ((self.Omega_Q - self.Q * self.Omega(v)) ** 2).sum() + self.lam * ((v - self.x0) ** 2).sum()

    def _get_params(self, verbose=True):

        self.result = minimize(self.objective, x0=self.x0, bounds=[(None, None), (None, None)] + [(-1, None)] * self.filter_function.ndim + [(None, None)] + [(-1, None)] * self.filter_function.ndim)

        if verbose:
            print(self.result)

        self.params = [self.result.x]

    def predict(self, verbose=True):
        if self.params is None:
            self._get_params(verbose=verbose)
        return self.Omega(self.params[0])

#

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    def run_tests(seed=0):

        np.random.seed(seed)

        # set variables
        N1 = 20
        N2 = 15
        N = N1 * N2

        graph = ProductGraph.lattice(N1, N2)
        signal = np.random.rand(N2, N1)
        signal[np.random.randint(0, 2, (N2, N1)).astype(bool)] = np.nan
        gamma = 0.02

        Y, S = get_y_and_s(signal)
        filter_function = UnivariateFilterFunction.diffusion(beta=1)
        G = filter_function(graph.lam)
        n_neighbours = graph.A.sum(0)

        X = np.array([np.ones(N),
                      vec(1 - S),
                      graph.U @ KroneckerDiag(G) @ graph.U.T @ vec(1 - S),
                      (graph.U ** 2) @ vec(G),
                      (graph.U ** 2) @ vec(G ** 2),
                      n_neighbours,
                      graph.U @ KroneckerDiag(G) @ graph.U.T @ n_neighbours
                      ]).T

        X[:, 1:] = X[:, 1:] / X[:, 1:].std(axis=0)
        X[:, 1:] = X[:, 1:] - X[:, 1:].mean(axis=0)

        # calculate the solution explicitly
        H2 = graph.U.to_array() @ diag(vec(G ** 2)) @ graph.U.to_array().T
        explicit_sigma = H2 @ np.linalg.inv(gamma * np.eye(N) + diag(vec(S)) @ H2)

        Omega_true = np.log(ten(diag(explicit_sigma), like=Y))

        Q = np.random.randint(0, 2, Omega_true.shape)
        Omega_Q = np.zeros_like(Omega_true)
        Omega_Q[Q.astype(bool)] = Omega_true[Q.astype(bool)]

        P = X.shape[1]

        def test_ridge_regression():

            lam = 0.5

            M = X.T @ diag(vec(Q)) @ X + lam * np.eye(X.shape[1])
            omega1 = ten(X @ solve(M, X.T @ vec(Omega_Q)), like=Q)

            estimator_rr = RRVarSolver(Omega_Q, Q, X, lam=lam)
            omega2 = estimator_rr.predict()

            assert np.allclose(omega1, omega2, atol=1e-3)


        def test_rnc_regression():

            lam = 0.5

            Hi2 = graph.U.to_array() @ diag(vec(G ** -2)) @ graph.U.to_array().T
            M = np.block([[diag(vec(Q)) + gamma * Hi2, diag(vec(Q)) @ X],
                          [X.T @ diag(vec(Q)),         X.T @ diag(vec(Q)) @ X + lam * np.eye(X.shape[1])]])

            om = np.concatenate([vec(Omega_Q), X.T @ vec(Omega_Q)])
            out = solve(M, om)
            omega1 = ten(out[:X.shape[0]] + X @ out[X.shape[0]:], like=Omega_Q)

            estimator_rnc = RNCVarSolver(Omega_Q, Q, X, graph, filter_function, gamma, lam=lam)
            omega2 = estimator_rnc.predict()

            assert np.allclose(omega1, omega2, atol=1e-3)


        def demonstrate_ridge_regression():

            fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(6, 8))

            axes[0, 0].imshow(Omega_true)
            axes[0, 0].set_title('Ω true')

            for ax, lam in zip(axes.flatten()[1:], np.logspace(-3, 1, 11)):

                estimator_rr =  RRVarSolver(Omega_Q, Q, X, lam=lam)
                ax.imshow(estimator_rr.predict(), vmin=Omega_true.min(), vmax=Omega_true.max())
                ax.set_title(f'λ = {lam:.3f}')

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle('Ridge Regression Estimates')

            plt.tight_layout()
            plt.show()


        def demonstrate_rnc_regression():

            fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(6, 8))

            axes[0, 0].imshow(Omega_true)
            axes[0, 0].set_title('Ω true')

            for ax, lam in zip(axes.flatten()[1:], np.logspace(-3, 1, 11)):

                estimator_rnc =  RNCVarSolver(Omega_Q, Q, X, graph, filter_function, gamma, lam=lam)
                ax.imshow(estimator_rnc.predict(), vmin=Omega_true.min(), vmax=Omega_true.max())
                ax.set_title(f'λ = {lam:.3f}')

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle('RNC Estimates')

            plt.tight_layout()
            plt.show()


        def demonstrate_lfp_regression():

            fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(6, 8))

            axes[0, 0].imshow(Omega_true)
            axes[0, 0].set_title('Ω true')

            for ax, lam in zip(axes.flatten()[1:], np.logspace(-3, 1, 11)):

                estimator_rnc =  LFPVarSolver(Omega_Q, Q, X, graph, filter_function, lam=lam)
                ax.imshow(estimator_rnc.predict(), vmin=Omega_true.min(), vmax=Omega_true.max())
                ax.set_title(f'λ = {lam:.3f}')

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle('LFP Estimates')

            plt.tight_layout()
            plt.show()


        def demonstrate_lfp_regression_mvf():

            fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(6, 8))

            axes[0, 0].imshow(Omega_true)
            axes[0, 0].set_title('Ω true')

            filter_function = MultivariateFilterFunction.diffusion(beta=np.array([1.0, 1.0]))

            for ax, lam in zip(axes.flatten()[1:], np.logspace(-3, 1, 11)):

                estimator_rnc =  LFPVarSolver(Omega_Q, Q, X, graph, filter_function, lam=lam)
                ax.imshow(estimator_rnc.predict(), vmin=Omega_true.min(), vmax=Omega_true.max())
                ax.set_title(f'λ = {lam:.3f}')

            for ax in axes.flatten():
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle('LFP Estimates, Multivariate Filter Function')

            plt.tight_layout()
            plt.show()

        test_ridge_regression()
        test_rnc_regression()

        demonstrate_ridge_regression()
        demonstrate_rnc_regression()
        demonstrate_lfp_regression()
        demonstrate_lfp_regression_mvf()

        print('All tests passed')

    run_tests()




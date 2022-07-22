import numpy as np
from numpy import ndarray, eye as I, diag
from typing import Union, Callable
import networkx as nx
from scipy.sparse import spmatrix

from graph.filters import FilterFunction, MultivariateFilterFunction
from graph.graphs import BaseGraph
from utils.checks import check_compatible
from utils.linalg import vec, ten
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
                 filter_function: FilterFunction,
                 gamma: float,
                 lam: float):

        super().__init__(Omega_Q, Q, X, lam)

        check_compatible(signal=Omega_Q, graph=graph, filter_function=filter_function)


        self.gamma = gamma
        self.N = np.prod(X.shape)
        self.filter_function = filter_function
        self.graph = graph

        if isinstance(self.filter_function, MultivariateFilterFunction):
            self.G = self.filter_function(self.graph.lams)

        else:
            self.G = self.filter_function(self.graph.lam)

        lamX, self.V = eigh(X[vec(Q)].T @ X[vec(Q)])
        self.DX = diag((lamX + lam) ** -0.5)

        self.PP = self.X @ self.V @ self.DX
        self.GPs = [self.G * (self.graph.GFT(self.Q * ten(self.PP[:, i], like=self.G))) for i in range(self.X.shape[1])]

    def A(self, x: ndarray):

        a = x[self.N:]
        A = ten(a, like=self.G)

        B1 = self.G * (self.graph.GFT(self.Q * self.graph.rGFT(self.G * A))) + self.gamma * A
        B2 = sum(a[i] * self.GPs[i] for i in range(self.X.shape[1]))
        b1 = np.array([(A * self.GPs[i]).sum() for i in range(self.X.shape[1])])

        return np.block([vec(B1 + B2), b1 + a])

    def Phi(self, x: ndarray):
        return np.block([vec(self.UN @ (self.G * ten(x[:self.N], like=self.G)) @ self.UT.T), self.V @ self.DX @ x[self.N:]])

    def PhiT(self, x: ndarray):
        return np.block([vec(self.G * (self.UN.T @ ten(x[:self.N], like=self.G) @ self.UT)), self.DX @ self.V.T @ x[self.N:]])

    def _get_params(self, verbose=False):

        y0 = np.block([vec(self.Omega_Q), self.X.T @ vec(self.Omega_Q)])
        out = conjugate_gradient(self.A, y0, Phi=self.Phi, PhiT=self.PhiT, verbose=verbose)
        self.params = [ten(out[:self.N * self.T], like=self.G), out[self.N * self.T:]]

    def predict(self, verbose=False):
        if self.params is None:
            self._get_params(verbose=verbose)
        return self.params[0] + ten(self.X @ self.params[1], like=self.G)


class LFPVarSolver(VarSolver):

    def __init__(self,
                 Omega_Q: ndarray,
                 Q: ndarray,
                 X: ndarray,
                 lam: float,
                 UT: ndarray,
                 UN: ndarray,
                 Lam: np.ndarray,
                 S_: ndarray,
                 A_: ndarray,
                 eta: Callable[[ndarray, float], ndarray],
                 beta0: float,
                 ):
        super().__init__(Omega_Q, Q, X, lam)

        self.UT = UT
        self.UN = UN
        self.Lam = Lam

        self.S_ = S_
        self.A_ = A_
        self.eta = eta

        self.x0 = np.array([0, 0, beta0, 0, beta0])

    def H(self, Y: ndarray, beta: float):
        """
        Apply the operation Y -> ten(H @ vec(Y)) efficiently for a filter defined by λ -> η(λ; β)
        """
        return self.UN @ (self.eta(self.Lam, beta) * (self.UN.T @ Y @ self.UT)) @ self.UT.T

    def Omega(self, v: ndarray):
        """
        Return the estitenor for Omega for a given objective vector
        """
        return v[0] + v[1] * self.H(self.S_, v[2]) + v[3] * self.H(self.A_, v[4])

    def objective(self, v: ndarray):
        """
        The objective function to minimise
        """
        return ((self.Omega_Q - self.Q * self.Omega(v)) ** 2).sum() + self.lam * ((v - self.x0) ** 2).sum()

    def _get_params(self, verbose=False):

        self.result = minimize(self.objective, x0=self.x0, bounds=[(None, None), (None, None), (-1, None), (None, None), (-1, None)])

        if verbose:
            print(self.result)

        self.params = [self.result.x]

    def predict(self, verbose=False):
        if self.params is None:
            self._get_params(verbose=verbose)
        return self.Omega(self.params[0])

#
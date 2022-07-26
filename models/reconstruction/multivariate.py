from algorithms.cgm import solve_SPCGM
from algorithms.query import select_Q_active
from algorithms.sim import solve_SIM
from graph.graphs import BaseGraph, ProductGraph
from graph.filters import _FilterFunction, UnivariateFilterFunction, MultivariateFilterFunction
from models.reconstruction.reconstruction_utils import get_y_and_s
from utils.checks import check_valid_graph, check_compatible
from kronecker.kron_base import KroneckerDiag
from utils.linalg import vec, ten
from models.reconstruction.var_estimators import RRVarSolver, LFPVarSolver, RNCVarSolver

from typing import Union
import numpy as np
from numpy import diag, ndarray
import networkx as nx


class MultivariateGraphSignalReconstructor:

    def __init__(self,
                 signal: ndarray,
                 graph: Union[BaseGraph, ndarray, nx.Graph],
                 filter_function: _FilterFunction,
                 gamma: float):

        self.Y, self.S = get_y_and_s(signal)

        # validate the graph and turn into a graph.Graph if not already
        self.graph = check_valid_graph(graph)
        self.filter_function = filter_function
        self.gamma = gamma

        # check the signal, graph and filter_function are all mutually compatible
        check_compatible(signal=self.Y, graph=self.graph, filter_function=self.filter_function)

        if isinstance(self.filter_function, MultivariateFilterFunction):
            self.G = self.filter_function(self.graph.lams)

        else:
            self.G = self.filter_function(self.graph.lam)

        n_neighbours = graph.A.sum(0)

        self.X = np.array([np.ones(graph.N),
                      vec(1 - self.S),
                      graph.U @ KroneckerDiag(self.G) @ graph.U.T @ vec(1 - self.S),
                      (graph.U ** 2) @ vec(self.G),
                      (graph.U ** 2) @ vec(self.G ** 2),
                      n_neighbours,
                      graph.U @ KroneckerDiag(self.G) @ graph.U.T @ n_neighbours
                      ]).T

    def set_gamma(self, gamma: float) -> 'MultivariateGraphSignalReconstructor':
        """
        Set the gamma parameter. Recompute M only.
        """
        self.gamma = gamma
        return self

    def set_beta(self, beta: Union[float, ndarray]) -> 'MultivariateGraphSignalReconstructor':
        """
        Set the beta parameter for the filter function. Recompute g2, H2 and M.
        """
        self.filter_function.set_beta(beta)
        return self

    def _compute_cgm(self, Y: ndarray, tol: float=1e-5,  verbose: bool=True) -> tuple[ndarray, int]:
        """
        Run the conjugate gradient method to compute the result of A^-1 vec(Y).
        """

        # mimic arrays, but use optimised KroneckerOperators
        DG = KroneckerDiag(self.G)
        DS = KroneckerDiag(self.S)
        C = DG @ self.graph.U.T @ DS @ self.graph.U @ DG
        A_precon = C + self.gamma * KroneckerDiag(np.ones_like(self.S))

        Phi = self.graph.U @ DG
        PhiT = Phi.T

        return solve_SPCGM(A_precon=A_precon,
                           y=Y,
                           Phi=Phi,
                           PhiT=PhiT,
                           reltol=tol,
                           verbose=verbose)

    def _compute_sim(self, Y: ndarray, tol: float=1e-5, verbose: bool=True) -> tuple[ndarray, int]:
        """
        Run the stationary iterative method to compute the result of A^-1 vec(Y).
        """

        J = self.G ** 2 / (self.G ** 2 + self.gamma)
        S_ = 1 - self.S
        Minv = self.graph.U @ KroneckerDiag(J) @ self.graph.U.T
        MinvN = Minv @ KroneckerDiag(S_)

        return solve_SIM(Y,
                         Minv=Minv,
                         MinvN=MinvN,
                         tol=tol,
                         verbose=verbose)

    def compute_mean(self, method='cgm', tol: float=1e-5, verbose: bool=False) -> ndarray:
        """
        Compute the reconstructed signal. `method` should be one of:
            * 'sim' for the Stationary Iterative Method
            * 'cgm' for the Conjugate Gradient Method
        """

        if method.lower() == 'sim':
            result, nits = self._compute_sim(self.Y, tol, verbose)

        elif method.lower() == 'cgm':
            result, nits = self._compute_cgm(self.Y, tol, verbose)

        else:
            raise ValueError('`method` should be "sim" or "cgm"')

        if verbose:
            print(f'{method.upper()} completed in {nits} iterations')

        return result

    def compute_logvar(self,
                       n_queries: int,
                       var_solver: str='lfp',
                       lam: float=0.001,
                       method='cgm',
                       tol=1e-5,
                       verbose: bool=True,
                       seed: int=0) -> ndarray:

        Q = select_Q_active(self.X, n_queries, shape=self.S.shape, seed=seed)
        Omega_Q = self.get_Omega_Q(Q, method=method, tol=tol, verbose=verbose)

        if var_solver.lower() == 'lfp':
            solver = LFPVarSolver(Omega_Q, Q, self.X, self.graph, self.filter_function, lam)

        elif var_solver.lower() == 'rr':
            solver = RRVarSolver(Omega_Q, Q, self.X, lam)

        elif var_solver.lower() == 'rnc':
            solver = RNCVarSolver(Omega_Q, Q, self.X, self.graph, self.filter_function, self.gamma, lam)

        else:
            raise ValueError('var_solver should be "lfp", "rr" or "rnc"')

        return solver.predict()


    def query(self, element: tuple, method='cgm', tol=1e-5, verbose: bool=False) -> float:
        """
        Directly compute the posterior variance at index `element`
        """

        Y = np.zeros_like(self.Y)
        Y[element] = 1

        if method.lower() == 'sim':
            result, nits = self._compute_sim(Y, tol, verbose)

        elif method.lower() == 'cgm':
            result, nits = self._compute_cgm(Y, tol, verbose)

        else:
            raise ValueError('`method` should be "sim" or "cgm"')

        return np.log(result[element])

    def get_Omega_Q(self, Q: ndarray, method='cgm', tol=1e-5, verbose: bool=True):

        om = np.zeros(self.S.shape)

        for element in np.argwhere(Q):
            element = tuple(element)
            om[element] = self.query(element, method=method, tol=tol, verbose=verbose)

        return om

    def compute_logvar_full(self, method='cgm', tol=1e-5, verbose: bool=True):
        return self.get_Omega_Q(np.ones_like(self.S), method=method, tol=tol, verbose=verbose)




if __name__ == '__main__':

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    def run_tests():

        # set variables
        # N1 = 5
        # N2 = 6
        # N3 = 7

        Ns = (20, 25)

        graph = ProductGraph.lattice(*Ns)
        signal = np.random.randn(*tuple(reversed(Ns)))
        signal[np.random.randint(0, 2, tuple(reversed(Ns))).astype(bool)] = np.nan
        gamma = 0.02

        Y, S = get_y_and_s(signal)
        filter_function = UnivariateFilterFunction.diffusion(beta=1)

        # calculate the solution explicitly
        H2 = graph.U.to_array() @ diag(filter_function(vec(graph.lam)) ** 2) @ graph.U.to_array().T
        explicit_sigma = np.linalg.inv(gamma * np.linalg.inv(H2) + diag(vec(S)))
        explicit_mean = ten(explicit_sigma @ vec(Y), like=Y)

        def test_mean():

            # calculate the solution
            reconstructor = MultivariateGraphSignalReconstructor(signal, graph, filter_function, gamma)
            reconstructor_mean1 = reconstructor.compute_mean(method='sim', tol=1e-8)
            reconstructor_mean2 = reconstructor.compute_mean(method='cgm', tol=1e-8)

            assert np.allclose(explicit_mean, reconstructor_mean1, atol=1e-5)
            assert np.allclose(explicit_mean, reconstructor_mean2, atol=1e-5)

        def test_omega():

            # calculate the solution
            reconstructor = MultivariateGraphSignalReconstructor(signal, graph, filter_function, gamma)
            reconstructor_omega1 = reconstructor.compute_Omega_full(method='sim', tol=1e-8)
            reconstructor_omega2 = reconstructor.compute_Omega_full(method='cgm', tol=1e-8)

            assert np.allclose(np.log(ten(diag(explicit_sigma), like=signal)), reconstructor_omega1, atol=1e-5)
            assert np.allclose(np.log(ten(diag(explicit_sigma), like=signal)), reconstructor_omega2, atol=1e-5)


        test_mean()
        test_omega()

        print('All tests passed')

    run_tests()






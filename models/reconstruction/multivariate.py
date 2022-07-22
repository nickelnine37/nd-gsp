from algorithms.cgm import solve_SPCGM
from algorithms.sim import solve_SIM
from graph.graphs import BaseGraph, Graph, ProductGraph
from graph.filters import FilterFunction, UnivariateFilterFunction, MultivariateFilterFunction
from models.reconstruction.reconstruction_utils import get_y_and_s
from utils.checks import check_valid_graph, check_compatible
from utils.kronecker import KroneckerDiag
from utils.linalg import vec, ten, vec_index, ten_index

from typing import Union
import numpy as np
from numpy import eye as I, diag, ndarray
import networkx as nx



class MultivariateGraphSignalReconstructor:

    def __init__(self,
                 signal: ndarray,
                 graph: Union[BaseGraph, ndarray, nx.Graph],
                 filter_function: FilterFunction,
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

        return solve_SPCGM(A_precon=lambda x: A_precon @ x,
                           y=Y,
                           Phi=lambda x: Phi @ x,
                           PhiT=lambda x: PhiT @ x,
                           reltol=tol,
                           verbose=verbose)

    def _compute_sim(self, Y: ndarray, tol: float=1e-5, verbose: bool=True) -> tuple[ndarray, int]:
        """
        Run the stationary iterative method to compute the result of A^-1 vec(Y).
        """

        J = self.G ** 2 / (self.G ** 2 + self.gamma)
        S_ = 1 - self.S

        return solve_SIM(Y,
                         Minv=lambda X: self.graph.scale_spectral(X, J),
                         MinvN=lambda X: self.graph.scale_spectral(S_ * X, J),
                         tol=tol,
                         verbose=verbose)


    def compute_mean(self, method='cgm', tol: float=1e-5, verbose: bool=True) -> ndarray:
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

    def estimate_var(self, method='cgm', tol=1e-5, verbose: bool=True) -> ndarray:

        def query(element: tuple):
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

            return result[element]





if __name__ == '__main__':

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    def run_tests():

        # set variables
        N1 = 5
        N2 = 6
        N3 = 7

        graph = ProductGraph.lattice(N1, N2, N3)
        signal = np.random.randn(N3, N2, N1)
        signal[np.random.randint(0, 2, (N3, N2, N1)).astype(bool)] = np.nan
        gamma = 0.02

        Y, S = get_y_and_s(signal)
        filter_function = UnivariateFilterFunction.diffusion(beta=1)

        def test_mean():


            # calculate the solution explicitly
            H2 = graph.U.to_array() @ diag(filter_function(vec(graph.lam)) ** 2) @ graph.U.to_array().T
            explicit_sigma = np.linalg.inv(gamma * np.linalg.inv(H2) + diag(vec(S)))
            explicit_mean = ten(explicit_sigma @ vec(Y), like=Y)

            # calculate the solution
            reconstructor = MultivariateGraphSignalReconstructor(signal, graph, filter_function, gamma)
            reconstructor_mean1 = reconstructor.compute_mean(method='sim', tol=1e-8)
            reconstructor_mean2 = reconstructor.compute_mean(method='cgm', tol=1e-8)

            assert np.allclose(explicit_mean, reconstructor_mean1, atol=1e-5)
            assert np.allclose(explicit_mean, reconstructor_mean2, atol=1e-5)


        test_mean()

        print('All tests passed')

    run_tests()






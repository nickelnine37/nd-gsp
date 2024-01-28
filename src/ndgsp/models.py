import numpy as np
import pandas as pd
import jax.numpy as jnp

from typing import List
from ndgsp.algorithms import *

from ndgsp.graphs import BaseGraph
from ndgsp.filters import FilterFunction
from ndgsp.utils.arrays import one_hot

from scipy.spatial.distance import squareform, pdist


class Processor:
    """
    Class to pre and post process inputs by converting between desired types 
    and optionally providing normalisation
    """
    
    def __init__(self, Y: np.ndarray, normalise=False):
        """
        Create a preprocessor for a given input Y. If normalise == True, remove global mean
        and divide by std. 
        """
        
        if isinstance(Y, np.ndarray):
            self.type = 'np'
            
        elif isinstance(Y, pd.DataFrame):
            self.type = 'pd'
            self.columns = Y.columns
            self.index = Y.index
            
        elif isinstance(Y, jnp.array):
            self.type = 'jax'
            
        else:
            raise ValueError(f'Expected type of Y to be np, pd or jax but it is {Y.type}')
        
        if normalise:
            
            Y = np.asarray(Y)
            self.loc = np.nanmean(Y)
            self.scale = np.nanmean(Y)
            
        else:
            
            self.loc = 0.0
            self.scale = 1.0
            
            
    def preprocess(self, Y: np.ndarray, one_hot_encode: bool=False):
        
        Y = np.asarray(Y).copy()

        if self.loc != 0 or self.scale != 1:
            Y -= self.loc
            Y /= self.scale
        
        S = ~np.isnan(Y)
        
        Y = np.nan_to_num(Y, copy=False, nan=0)
        
        if one_hot_encode:
            
            Y = one_hot(Y.astype(int), num_classes=int(Y.max()) + 1)
            Y *= S.astype(int)[..., None]
        
        return Y, S
    
    
    def postprocess(self, F: np.ndarray, one_hot_decode: bool=False):

        if one_hot_decode:
            F = np.argmax(F, axis=-1)
        
        if self.type == 'np':
            return np.asarray(F) * self.scale + self.loc
            
        elif self.type == 'pd':
            return pd.DataFrame(np.asarray(F) * self.scale + self.loc, index=self.index, columns=self.columns)
            
        elif self.type == 'jax':
            return jnp.asarray(F) * self.scale + self.loc


class Model:
    
    def __init__(self, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float):

        self.graph = graph
        self.set_gamma(gamma)
        self.filter = filter_func
        self.U = self.graph.U
        self.G = self.graph.get_G(self.filter)
        
    def set_gamma(self, gamma: float):
        assert gamma > 0, f'gamma must be greater than zero but it is {gamma}'
        self.gamma = gamma
        return self
        
    def set_beta(self, beta):

        if hasattr(beta, '__iter__'):
            assert all(b > 0 for b in beta), f'beta must be greater than zero but it is {beta}'

        else:
            assert beta > 0, f'beta must be greater than zero but it is {beta}'

        self.filter.set_beta(beta)
        self.G = self.graph.get_G(self.filter)
        return self

        
class ReconstructionModel(Model):
    
    def __init__(self, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float):
        super().__init__(Y, graph, filter_func, gamma)
        

class KernelModel(Model):

    def __init__(self, X: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, kernel_std: float):
        super().__init__(Y, graph, filter_func, gamma)

        assert X.ndim == 2, f'X should be 2d but it has {X.ndim} dims (shape {X.shape})'        
        assert X.shape[0] == Y.shape[0], f'X and Y should have the same length first dim, but they have shape {X.shape[0]} and {Y.shape[0]} respectively'

        self.X = X
        self.D = squareform(pdist(X, metric='sqeuclidean'))
        self.set_kernel_std(kernel_std)

    def set_kernel_std(self, kernel_std: float):
        self.kernel_std = kernel_std
        self.K = np.exp(-0.5 * self.D / kernel_std ** 2)
        self.lamK, self.V = np.linalg.eigh(self.K)
        return self



class CohesionModel(Model):

    def __init__(self, X: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, lam: float):
        super().__init__(Y, graph, filter_func, gamma)

        self.X = X
        self.set_lam(lam)

    def set_lam(self, lam: float):

        assert lam >= 0, f'lam should be greater than or equal to zero but it is {lam}'
        self.lam = lam


class KernelCohesionModel(Model):

    def __init__(self, X: np.ndarray, X_: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, lam: float, kernel_std: float):
        super().__init__(Y, graph, filter_func, gamma)

        assert X_.ndim == 2, f'X should be 2d but it has {X.ndim} dims (shape {X.shape})'        
        assert X_.shape[0] == Y.shape[0], f'X and Y should have the same length first dim, but they have shape {X.shape[0]} and {Y.shape[0]} respectively'

        self.X_ = X_
        self.X = X
        self.D = squareform(pdist(X_, metric='sqeuclidean'))
        self.set_kernel_std(kernel_std)
        self.set_lam(lam)

    def set_kernel_std(self, kernel_std: float):
        self.kernel_std = kernel_std
        self.K = np.exp(-0.5 * self.D / kernel_std ** 2)
        self.lamK, self.V = np.linalg.eigh(self.K)
        return self
    
    def set_lam(self, lam: float):

        assert lam >= 0, f'lam should be greater than or equal to zero but it is {lam}'
        self.lam = lam



class GSR(ReconstructionModel):
    
    def __init__(self, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float):
        super().__init__(Y, graph, filter_func, gamma)
        
        self.processor = Processor(Y, normalise=True)
        self.Y, self.S = self.processor.preprocess(Y)
        
        assert self.Y.shape == self.graph.signal_shape, f'Y and graph should have compatible dims but they have {self.Y.shape} and {self.graph.signal_shape} respectively'
        
    def solve(self) -> np.ndarray:
        """
        Compute the posterior mean F
        """
        
        F = solve_gsr(self.Y, self.S, self.U, self.G, self.gamma)
        
        return self.processor.postprocess(F)
    
    def sample(self, n_samples: int=1, seed: int=None) -> List[np.ndarray]:
        """
        Draw samples from the posterior distribution
        """
        
        Fs = sample_gsr(self.Y, self.S, self.U, self.G, self.gamma, n_samples, seed)
    
        return [self.processor.postprocess(F) for F in Fs] 

    def estimate_marginal_variance(n_samples: int, mean: np.ndarray=None, seed: int=None) -> np.ndarray:
        """
        Estimate the marginal variance using a memory efficient generator pattern
        """
        
        if mean is None:
            mean = self.solve()

        var = np.zeros_like(mean)
        
        for k, sample in enumerate(generate_samples_gsr(self.Y, self.S, self.U, self.G, self.gamma, n_samples, seed)):
            
            var = (k * var + (self.processor.postprocess(sample) - mean) ** 2) / (k + 1)

        return var


class LGSR(ReconstructionModel):
    
    def __init__(self, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float):
        super().__init__(Y, graph, filter_func, gamma)

        self.processor = Processor(Y, normalise=False)
        self.Y, self.S = self.processor.preprocess(Y)
        
        assert self.Y.shape == self.graph.signal_shape, f'Y and graph should have compatible dims but they have {self.Y.shape} and {self.graph.signal_shape} respectively'
        
        
    def solve(self, classify: bool=False) -> np.ndarray:
        """
        Compute the posterior mean mu(F)
        """
        
        F = solve_lgsr(self.Y, self.S, self.U, self.G, self.gamma)
        
        if classify:
            return self.processor.postprocess((F > 0).astype(int)) 

        else:
            return self.processor.postprocess(mu_logistic(F)) 
        
    
class MulticlassLGSR(ReconstructionModel):
    
    def __init__(self, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float):
        super().__init__(Y, graph, filter_func, gamma)

        self.processor = Processor(Y, normalise=False)
        self.Y, self.S = self.processor.preprocess(Y, one_hot_encode=True)
        
        assert self.Y.shape[:-1] == self.graph.signal_shape, f'Y and graph should have compatible dims but they have {self.Y.shape[:-1]} and {self.graph.signal_shape} respectively'

    def solve(self, classify: bool=False) -> np.ndarray:
        """
        Compute the posterior mean mu(F)
        """
        
        F = solve_lgsr_multiclass(self.Y, self.S, self.U, self.G, self.gamma)
        
        if classify:
            return self.processor.postprocess(F, one_hot_decode=True) 

        else:
            return self.processor.postprocess(mu_softmax(F), one_hot_decode=False) 



class KGR(KernelModel):

    def __init__(self, X: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, kernel_std: float):
        super().__init__(X, Y, graph, filter_func, gamma, kernel_std)

        self.processor = Processor(Y, normalise=True)
        self.Y, self.S = self.processor.preprocess(Y)
        
        assert self.Y.shape[1:] == self.graph.signal_shape, f'Y and graph should have compatible dims but they have {self.Y.shape} and {self.graph.signal_shape} respectively'

    def solve(self) -> np.ndarray:
        """
        Compute the posterior mean F
        """
        
        F = solve_kgr(self.Y, self.S, self.U, self.G, self.V, self.lamK, self.gamma)
        
        return self.processor.postprocess(F)

    def sample(self, n_samples: int=1, seed: int=None) -> List[np.ndarray]:
        """
        Draw samples from the posterior distribution
        """
        
        Fs = sample_kgr(self.Y, self.S, self.U, self.G, self.V, self.lamK, self.gamma, n_samples, seed)
    
        return [self.processor.postprocess(F) for F in Fs]
    
    def estimate_marginal_variance(n_samples: int, mean: np.ndarray=None, seed: int=None) -> np.ndarray:
        """
        Estimate the marginal variance using a memory efficient generator pattern
        """
        
        if mean is None:
            mean = self.solve()

        var = np.zeros_like(mean)
        
        for k, sample in enumerate(generate_samples_kgr(self.Y, self.S, self.U, self.G, self.V, self.lamK, self.gamma, n_samples, seed)):
            
            var = (k * var + (self.processor.postprocess(sample) - mean) ** 2) / (k + 1)

        return var
    

class LKGR(KernelModel):

    def __init__(self, X: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, kernel_std: float):
        super().__init__(X, Y, graph, filter_func, gamma, kernel_std)

        self.processor = Processor(Y, normalise=False)
        self.Y, self.S = self.processor.preprocess(Y)
        
        assert self.Y.shape[1:] == self.graph.signal_shape, f'Y and graph should have compatible dims but they have {self.Y.shape} and {self.graph.signal_shape} respectively'
        self.N = np.prod(Y.shape)


    def solve(self, classify: bool=False) -> np.ndarray:
        """
        Compute the posterior mean mu(F)
        """
        
        F = solve_lkgr(self.Y, self.S, self.U, self.G, self.V, self.lamK, self.gamma)
        
        if classify:
            return self.processor.postprocess((F > 0).astype(int)) 

        else:
            return self.processor.postprocess(mu_logistic(F)) 
        

class MulticlassLKGR(KernelModel):

    def __init__(self, X: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, kernel_std: float):
        super().__init__(X, Y, graph, filter_func, gamma, kernel_std)

        self.processor = Processor(Y, normalise=False)
        self.Y, self.S = self.processor.preprocess(Y)
        
        assert self.Y.shape[1:-1] == self.graph.signal_shape, f'Y and graph should have compatible dims but they have {self.Y.shape} and {self.graph.signal_shape} respectively'

    def solve(self, classify: bool=False) -> np.ndarray:
        """
        Compute the posterior mean mu(F)
        """
        
        F = solve_lkgr_multiclass(self.Y, self.S, self.U, self.G, self.V, self.lamK, self.gamma)
        
        if classify:
            return self.processor.postprocess(F, one_hot_decode=True) 

        else:
            return self.processor.postprocess(mu_softmax(F), one_hot_decode=False) 
            


class RNC(CohesionModel):

    def __init__(self, X: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, lam: float):
        super().__init__(X, Y, graph, filter_func, gamma, lam)

        self.processor = Processor(Y, normalise=True)
        self.Y, self.S = self.processor.preprocess(Y, one_hot_encode=False)

        assert X.shape[:-1] == Y.shape, f'X, and Y do not have compatible dimensions: {X.shape} and {Y.shape} respectively' 
        self.N = np.prod(Y.shape)
        self.M = X.shape[-1]


    def solve(self) -> np.ndarray:
        """
        Compute the posterior mean F
        """
        
        theta = solve_rnc(self.X, self.Y, self.S, self.U, self.G, self.gamma, self.lam)

        self.B = theta[:self.N].reshape(self.Y.shape)
        self.w = theta[self.N:]

        F = self.B + (self.X.reshape(-1, self.M) @ self.w).reshape(self.Y.shape)

        return self.processor.postprocess(F)
    
    def sample(self, n_samples: int=1, seed: int=None) -> List[np.ndarray]:
        """
        Draw samples from the posterior distribution
        """
        
        thetas = sample_rnc(self.X, self.Y, self.S, self.U, self.G, self.gamma, self.lam, n_samples, seed)

        def f(theta):
            return (theta[:self.N] + self.X.reshape(-1, self.M) @ theta[self.N:]).reshape(self.Y.shape)
        
        Fs = [f(theta) for theta in thetas]
    
        return [self.processor.postprocess(F) for F in Fs] 
    
    def estimate_marginal_variance(n_samples: int, mean: np.ndarray=None, seed: int=None) -> np.ndarray:
        """
        Estimate the marginal variance using a memory efficient generator pattern
        """

        def f(theta):
            return (theta[:self.N] + self.X.reshape(-1, self.M) @ theta[self.N:]).reshape(self.Y.shape)
        
        if mean is None:
            mean = self.solve()

        var = np.zeros_like(mean)
        
        for k, sample in enumerate(generate_samples_rnc(self.X, self.Y, self.S, self.U, self.G, self.gamma, self.lam, n_samples, seed)):

            F = f(sample)
            
            var = (k * var + (self.processor.postprocess(F) - mean) ** 2) / (k + 1)

        return var
    
        
class LRNC(CohesionModel):

    def __init__(self, X: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, lam: float, kernel_std: float):
        super().__init__(X, Y, graph, filter_func, gamma, lam)

        self.processor = Processor(Y, normalise=False)
        self.Y, self.S = self.processor.preprocess(Y, one_hot_encode=False)

        assert X.shape[:-1] == Y.shape, f'X, and Y do not have compatible dimensions: {X.shape} and {Y.shape} respectively' 
        self.N = np.prod(Y.shape)
        self.M = X.shape[-1]

    def F(self, theta):
        return (theta[:self.N] + self.X.reshape(-1, self.M) @ theta[self.N:]).reshape(self.Y.shape)
    
    def solve(self, classify: bool=False) -> np.ndarray:
        """
        Compute the posterior mean F
        """
        
        theta = solve_lrnc(self.X, self.Y, self.S, self.U, self.G, self.gamma, self.lam)

        self.B = theta[:self.N].reshape(self.Y.shape)
        self.w = theta[self.N:]

        F = self.F(theta)

        if classify:
            return self.processor.postprocess((F > 0).astype(int)) 

        else:
            return self.processor.postprocess(mu_logistic(F)) 


class MulticlassLRNC(CohesionModel):

    def __init__(self, X: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, lam: float):
        super().__init__(X, Y, graph, filter_func, gamma, lam)

        assert X.shape[:-1] == Y.shape, f'X, and Y do not have compatible dimensions: {X.shape} and {Y.shape} respectively' 

        self.processor = Processor(Y, normalise=False)
        self.Y, self.S = self.processor.preprocess(Y, one_hot_encode=True)

        self.N = np.prod(self.Y.shape[:-1])
        self.M = X.shape[-1]
        self.C = self.Y.shape[-1]

    def F(self, theta):
        return theta[:self.N * self.C].reshape(self.Y.shape) + (self.X @ theta[self.N * self.C:].reshape(self.M, self.C)).reshape(self.Y.shape)
    
    def solve(self, classify: bool=False) -> np.ndarray:
        """
        Compute the posterior mean F
        """
        
        theta = solve_lrnc_multiclass(self.X, self.Y, self.S, self.U, self.G, self.gamma, self.lam)

        F = self.F(theta)

        if classify:
            return self.processor.postprocess(F, one_hot_decode=True) 

        else:
            return self.processor.postprocess(mu_softmax(F), one_hot_decode=False) 
        


class KGRNC(KernelCohesionModel):

    def __init__(self, X: np.ndarray, X_: np.ndarray, Y: np.ndarray, graph: BaseGraph, filter_func: FilterFunction, gamma: float, lam: float, kernel_std: float):
        super().__init__(X, X_, Y, graph, filter_func, gamma, lam, kernel_std)

        self.processor = Processor(Y, normalise=True)
        self.Y, self.S = self.processor.preprocess(Y, one_hot_encode=False)

        assert X.shape[:-1] == Y.shape, f'X, and Y do not have compatible dimensions: {X.shape} and {Y.shape} respectively' 
        self.N = np.prod(Y.shape)
        self.M = X.shape[-1]


    def solve(self) -> np.ndarray:
        """
        Compute the posterior mean F
        """
        
        theta = solve_kgrnc(self.X, self.Y, self.S, self.U, self.G, self.V, self.lamK, self.gamma, self.lam)

        self.B = theta[:self.N].reshape(self.Y.shape)
        self.w = theta[self.N:]

        F = self.B + (self.X.reshape(-1, self.M) @ self.w).reshape(self.Y.shape)

        return self.processor.postprocess(F)
    

    def sample(self, n_samples: int=1, seed: int=None) -> List[np.ndarray]:
        """
        Draw samples from the posterior distribution
        """
        
        thetas = sample_kgrnc(self.X, self.Y, self.S, self.U, self.G, self.V, self.lamK, self.gamma, self.lam, n_samples, seed)

        def f(theta):
            return (theta[:self.N] + self.X.reshape(-1, self.M) @ theta[self.N:]).reshape(self.Y.shape)
        
        Fs = [f(theta) for theta in thetas]
    
        return [self.processor.postprocess(F) for F in Fs] 
    
    def estimate_marginal_variance(n_samples: int, mean: np.ndarray=None, seed: int=None) -> np.ndarray:
        """
        Estimate the marginal variance using a memory efficient generator pattern
        """

        def f(theta):
            return (theta[:self.N] + self.X.reshape(-1, self.M) @ theta[self.N:]).reshape(self.Y.shape)
        
        if mean is None:
            mean = self.solve()

        var = np.zeros_like(mean)
        
        for k, sample in enumerate(generate_samples_kgrnc(self.X, self.Y, self.S, self.U, self.G, self.V, self.lamK, self.gamma, self.lam, n_samples, seed)):

            F = f(sample)
            
            var = (k * var + (self.processor.postprocess(F) - mean) ** 2) / (k + 1)

        return var
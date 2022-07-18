from numpy import ndarray, exp
import numpy as np
from typing import Callable


class FilterFunction: 
    
    def __init__(self, filter: Callable, beta: float):
        self.filter = filter
        self.beta = beta

    @classmethod
    def random_walk(cls, beta: ndarray):
        raise NotImplementedError

    @classmethod
    def diffusion(cls, beta: ndarray):
        raise NotImplementedError

    @classmethod
    def ReLu(cls, beta: ndarray):
        raise NotImplementedError

    @classmethod
    def sigmoid(cls, beta: ndarray):
        raise NotImplementedError

    @classmethod
    def bandlimited(cls, beta: ndarray):
        raise NotImplementedError

    def set_beta(self, beta: ndarray):
        raise NotImplementedError

    def __call__(self, Lams: ndarray):
        raise NotImplementedError


class UnivariateFilterFunction(FilterFunction):
    """
    Class to represent a simple graph filter. Initialise with one of the constructors:

        * random_walk
        * diffusion
        * ReLu
        * sigmoid
        * bandlimited

    each of which take a single beta float parameter. Aliternatively, can be initialised
    with a custom graph filter function.

    Examples:

    # ------- use with builtin filter -------

    fil = GraphFilter.random_walk(beta=1.5)

    fil(2)
    >>> 0.25

    fil(np.random.uniform(0, 4, (4, 4)))
    >>> [[0.258 0.165 0.253 0.278]
         [0.154 0.249 0.506 0.181]
         [0.17  0.334 0.694 0.305]
         [0.391 0.439 0.226 0.518]]

    # ------- use with custom filter --------

    def two_hop_random_walk(lam, beta):
        return (1 + beta * lam) ** -2

    fil = GraphFilter(two_hop_random_walk, beta=1)

    fil(3)
    >>> 0.0625

    """

    def __init__(self, filter: Callable, beta: float):
        super().__init__(filter, beta)


    @classmethod
    def random_walk(cls, beta: float):
        return cls(lambda lam, bet: (1 + bet * lam) ** -1, beta)

    @classmethod
    def diffusion(cls, beta: float):
        return cls(lambda lam, bet: exp(-bet * lam), beta)

    @classmethod
    def ReLu(cls, beta: float):
        return cls(lambda lam, bet: np.maximum(1 - bet * lam, 0), beta)

    @classmethod
    def sigmoid(cls, beta: float):
        return cls(lambda lam, bet: 2 * exp(-bet * lam) * (1 + exp(-bet * lam)) ** -1, beta)

    @classmethod
    def bandlimited(cls, beta: float):
        return cls(lambda lam, bet: (lam <= 1 / bet).astype(float) if bet != 0 else np.ones_like(lam), beta)

    def set_beta(self, beta: float):
        self.beta = beta

    def __call__(self, lam: ndarray):
        return self.filter(lam, self.beta)


class MultivariateFilterFunction(FilterFunction):
    """
    This slightly more complex filter type can handle different parameters in
    different dimensions. We have  list of parameters, betas, and a list of
    inputs, lams, then we filter as a function of betas.T @ lams.

    betas should have shape (M, ). Lams should have shape (M, N1, N2, ...)

    Example:

    fil = SpaceTimeGraphFilter.random_walk(betas=[1, 2, 3])

    lams = [np.random.randn(2, 2, 2) for i in range(3)]
    fil(lams)

    >>> array([[[ 1.133,  0.57 ],
                [-0.624,  0.447]],

               [[-0.64 ,  0.161],
                [ 0.279, -0.273]]])

    """

    def __init__(self, filter: Callable, beta: ndarray):
        self.filter = filter
        self.betas = beta
        self.n = len(beta)

    @classmethod
    def random_walk(cls, beta: ndarray):
        return cls(lambda Lams, betas_: (1 + sum(beta * Lam for beta, Lam in zip(betas_, Lams))) ** -1, beta)

    @classmethod
    def diffusion(cls, beta: ndarray):
        return cls(lambda Lams, betas_: exp(- sum(beta * Lam for beta, Lam in zip(betas_, Lams))), beta)

    @classmethod
    def ReLu(cls, beta: ndarray):
        return cls(lambda Lams, betas_: np.maximum(1 - sum(beta * Lam for beta, Lam in zip(betas_, Lams)), 0), beta)

    @classmethod
    def sigmoid(cls, beta: ndarray):
        def fil(Lams, betas_):
            E = exp(-sum(beta * Lam for beta, Lam in zip(betas_, Lams)))
            return 2 * E * (1 + E) ** -1

        return cls(fil, beta)

    @classmethod
    def bandlimited(cls, beta: ndarray):
        return cls(lambda Lams, betas_: np.all([beta * Lam < 1 for beta, Lam in zip(betas_, Lams)], axis=0).astype(float), beta)

    def set_beta(self, beta: ndarray):
        assert len(beta) == self.n, f'beta should be length {self.n} but it is length {len(beta)}'
        self.betas = beta

    def __call__(self, Lams: ndarray):
        assert len(Lams) == self.n, f'Lams should be length {self.n} but it is length {len(Lams)}'
        return self.filter(Lams, self.betas)

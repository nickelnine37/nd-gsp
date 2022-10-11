from ndgsp.graph.filters import UnivariateFilterFunction, MultivariateFilterFunction
import numpy as np


def test_univriate():
    lam = np.exp(-np.random.normal(size=10))

    for f in [UnivariateFilterFunction.random_walk(1),
              UnivariateFilterFunction.diffusion(1),
              UnivariateFilterFunction.ReLu(1),
              UnivariateFilterFunction.sigmoid(1),
              UnivariateFilterFunction.bandlimited(1)]:
        f.set_beta(2)
        f(lam)


def test_multivariate():

    lam = np.exp(-np.random.normal(size=(2, 10, 10)))

    for f in [MultivariateFilterFunction.random_walk([1, 1]),
              MultivariateFilterFunction.diffusion([1, 1]),
              MultivariateFilterFunction.ReLu([1, 1]),
              MultivariateFilterFunction.sigmoid([1, 1]),
              MultivariateFilterFunction.bandlimited([1, 1])]:

        f.set_beta([2, 2])

        f(lam)

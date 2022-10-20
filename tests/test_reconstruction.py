import numpy as np
from pykronecker import KroneckerDiag, KroneckerIdentity

from ndgsp.graph.filters import MultivariateFilterFunction
from ndgsp.graph.graphs import ProductGraph
from ndgsp.models.reconstruction import GSR, LogisticGSR

np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)


def test_real_reconstruction():
    """
    Test the real reconstruction model GSR. First, test that compute_mean() returns the right answer
    as computed using literal inversion of the linear system. Next, check that the sample covariance
    matrix is close to the real covariance matrix.
    """

    N1 = 10
    N2 = 12

    graph = ProductGraph.lattice(N1, N2)
    filter_func = MultivariateFilterFunction.diffusion(beta=[0.1, 0.1])

    Y = graph.filter(np.random.normal(size=(N1, N2)), filter_func)

    S = np.random.randint(0, 2, size=(N1, N2))

    YY = Y.at[~S.astype(bool)].set(np.nan)

    Y = Y.at[~S.astype(bool)].set(0)

    model = GSR(YY, graph, filter_func, gamma=1)

    Y_mod = model.compute_mean()

    Hi2 = graph.U @ KroneckerDiag(graph.get_G(filter_func) ** -2) @ graph.U.T

    A = np.linalg.inv((Hi2 + KroneckerDiag(S)).to_array())

    Y_mod2 = (A @ Y.reshape(-1)).reshape(Y.shape)

    assert np.allclose(Y_mod, Y_mod2, atol=1e-3, rtol=1e-4)

    assert ((A - np.cov([s.reshape(-1) for s in model.sample(100)], rowvar=False)) ** 2).sum() < 35


def test_logistic_reconstruction():
    """
    Test the logistic reconstruction model. First, test that the gradient of the objective function
    is zero at the computed mean. Next test that the samples have the correct covariance.
    """

    N1 = 10
    N2 = 12

    graph = ProductGraph.lattice(N1, N2)
    filter_func = MultivariateFilterFunction.diffusion(beta=[0.1, 0.1])

    Y = (graph.filter(np.random.normal(size=(N1, N2)), filter_func) > 0.5).astype(float)

    S = np.random.randint(0, 2, size=(N1, N2))

    YY = Y.at[~S.astype(bool)].set(np.nan)

    Y = Y.at[~S.astype(bool)].set(0)

    model = LogisticGSR(YY, graph, filter_func, gamma=1)

    DG = KroneckerDiag(graph.get_G(filter_func))
    DS = KroneckerDiag(S)
    alpha_star = model._compute_alpha_star()
    mu = model.get_mu(graph.U @ DG @ alpha_star)

    grad = DG @ graph.U.T @ DS @ (mu - Y) + alpha_star

    assert np.allclose(grad, 0, atol=1e-3, rtol=1e-2)

    def mu_inv(mu):
        return -np.log(mu ** -1 - 1)

    Dmu_ = KroneckerDiag(mu * (1 - mu))

    H = DG @ graph.U.T @ DS @ Dmu_ @ DS @ graph.U @ DG + KroneckerIdentity(like=DG)

    DGi = DG.inv()
    cov = np.cov([(DGi @ graph.U.T @ mu_inv(sample)).reshape(-1) for sample in model.sample(100)], rowvar=False)

    assert ((np.linalg.inv(H.to_array()) - cov) ** 2).sum() < 150



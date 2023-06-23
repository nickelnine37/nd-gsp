import numpy as np
import jax.numpy as jnp
import networkx as nx
from ndgsp import ProductGraph, MultivariateFilterFunction
from ndgsp.utils.arrays import one_hot
from scipy.spatial.distance import squareform, pdist


def get_test_data(seed=0):

    np.random.seed(seed)
    
    T = 3
    N1 = 5
    N2 = 6
    C = 3
    M = 4

    g = ProductGraph.lattice(N1, N2)
    fil = MultivariateFilterFunction.diffusion([0.3, 0.4])
    
    indsx, indsy = np.indices((N1, N2)).reshape(2, -1).T[np.random.choice(N1 * N2, size=int(N1 * N2 / 2), replace=False)].T
    S = np.ones((N1, N2))
    S[indsx, indsy] = 0
    
    indsx, indsy, indsz = np.indices((T, N1, N2)).reshape(3, -1).T[np.random.choice(T * N1 * N2, size=int(T * N1 * N2 / 2), replace=False)].T
    S_kgr = np.ones((T, N1, N2))
    S_kgr[indsx, indsy, indsz] = 0
    
    U = g.U
    G = g.get_G(fil)
    
    X = np.random.normal(size=(N1, N2, M))
    
    Y = S * np.random.normal(size=(N1, N2))
    Y_logistic = S * np.random.randint(2, size=(N1, N2)).astype(float)
    Y_multiclass = S[..., None] * one_hot(np.random.randint(C, size=(N1, N2)), C).astype(float)
    
    x = np.random.normal(size=(T, 5))
    lamK, V = jnp.linalg.eigh(np.exp(-squareform(pdist(x, metric='sqeuclidean'))))
    
    Y_kgr = S_kgr * np.random.normal(size=(T, N1, N2))
    Y_kgr_logistic = S_kgr * np.random.randint(2, size=(T, N1, N2)).astype(float)
    Y_kgr_multiclass = S_kgr[..., None] * np.random.normal(size=(T, N1, N2, C))
    
    gamma = 1.2
    lam = 5

    return X, Y, Y_logistic, Y_multiclass, Y_kgr, Y_kgr_logistic, Y_kgr_multiclass, S, S_kgr, U, G, gamma, lam, lamK, V

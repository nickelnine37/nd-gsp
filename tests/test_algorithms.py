from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import jit
from pykronecker import KroneckerBlockDiag, KroneckerProduct, KroneckerDiag

from ndgsp.utils.arrays import outer_product
from ndgsp.operators import KroneckerExpanded
from utils import get_test_data
from ndgsp.algorithms import *


def grad_gsr(F: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gradient of the nll for gsr evaluated at a point F
    """
    Y = jnp.asarray(Y)
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DGi2 = KroneckerDiag(G ** -2)
    DS = KroneckerDiag(S)
    return np.asarray(DS @ (F - Y) + gamma * U @ DGi2 @ U.T @ F)


def grad_kgr(F: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, V: np.ndarray, lamK: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gradient of the nll for kgr evaluated at a point F
    """
    Y = jnp.asarray(Y)
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DS = KroneckerDiag(S)
    
    V = jnp.asarray(V)
    lamK = jnp.asarray(lamK)
    
    U_ = KroneckerProduct([V] + U.As)
    G_ = outer_product(lamK ** 0.5, G)
    
    DGi2 = KroneckerDiag(G_ ** -2)
    
    return np.asarray(DS @ (F - Y) + gamma * U_ @ DGi2 @ U_.T @ F)


def grad_rnc(theta: np.ndarray, X: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """
    Gradient of the nll for gsr evaluated at a point theta
    """
    
    Y = jnp.asarray(Y)
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DGi2 = KroneckerDiag(G ** -2)
    DS = KroneckerDiag(S)
    X = jnp.asarray(X.reshape(-1, X.shape[-1]))
            
    N, M = X.shape
    X_ = jnp.concatenate([jnp.eye(N), X], axis=1)
    DS_ = jnp.concatenate([DS.to_array(), X.T @ DS], axis=0)
        
    return np.asarray(DS_ @ (X_ @ theta - Y.reshape(-1)) + KroneckerBlockDiag([gamma * U @ DGi2 @ U.T, lam * np.eye(M)]) @ theta)



def grad_lgsr(F: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gradient of nll of gsr model evaluated at F
    """
    Y = jnp.asarray(Y.astype(float))
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DGi2 = KroneckerDiag(G ** -2)
    DS = KroneckerDiag(S)
    
    @jit
    def mu(F):
        return 1 / (1 + jnp.exp(-F))
    
    return np.asarray(DS @ (mu(F)- Y) + gamma * U @ DGi2 @ U.T @ F)
    

def grad_lkgr(F: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, V: np.ndarray, lamK: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gradient of nll of lkgr model evaluated at F
    """
    
    V = jnp.asarray(V)
    lamK = jnp.asarray(lamK)
    
    U_ = KroneckerProduct([V] + U.As)
    G_ = outer_product(lamK ** 0.5, G)
    
    DGi2 = KroneckerDiag(G_ ** -2)
    DS = KroneckerDiag(S)
    
    @jit
    def mu(F):
        return 1 / (1 + jnp.exp(-F))
    
    return np.asarray(DS @ (mu(F)- Y) + gamma * U_ @ DGi2 @ U_.T @ F)


def grad_lrnc(theta: np.ndarray, X: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """
    Evaluate the gradient of the nll of the lrnc model at a point theta
    """
    
    Y = jnp.asarray(Y.astype(float))
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DGi2 = KroneckerDiag(G ** -2)
    DS = KroneckerDiag(S)
    X = jnp.asarray(X.reshape(-1, X.shape[-1]))
    
    N = np.prod(Y.shape)
    M = X.shape[-1]

    @jit
    def f(theta):
        return (theta[:N] + X @ theta[N:]).reshape(Y.shape)
    
    @jit
    def mu(theta):
        return 1 / (1 + jnp.exp(-f(theta)))
    
    Muy = (mu(theta) - Y).ravel()
    
    return np.asarray(jnp.concatenate([DS @ Muy, X.T @ DS @ Muy]) + KroneckerBlockDiag([gamma * U @ DGi2 @ U.T, lam * np.eye(M)]) @ theta)



def grad_lgsr_multiclass(F: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gradient of nll of multiclass lgsr model evaluated at a point F
    """
    
    C = Y.shape[-1]
    
    Y = jnp.asarray(Y.astype(float))
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DS_ = KroneckerExpanded(KroneckerDiag(S), C)
    DGi2 = KroneckerDiag(G ** -2)

    @jit
    def mu(F):
        F_ = jnp.exp(F)
        return F_ / F_.sum(-1)[..., None] 
    
    return DS_ @ (mu(F) - Y) + gamma * KroneckerExpanded(U @ DGi2 @ U.T, C) @ F


def grad_lkgr_multiclass(F: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, V: np.ndarray, lamK: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gradient of nll of multiclass lkgr model evaluated at a point F
    """
    
    V = jnp.asarray(V)
    lamK = jnp.asarray(lamK)
    
    U_ = KroneckerProduct([V] + U.As)
    G_ = outer_product(lamK ** 0.5, G)
    
    return grad_lgsr_multiclass(F, Y, S, U_, G_, gamma)


def grad_lrnc_multiclass(theta: np.ndarray, X: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """
    Gradient of the nll of the lrnc multiclass model evaluateted at theta
    """
    
    N = np.prod(G.shape)
    M = X.shape[-1]
    C = Y.shape[-1]
    
    Y = jnp.asarray(Y.astype(float))
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DGi2 = KroneckerDiag(G ** -2)
    DS = KroneckerDiag(S)
    X = jnp.asarray(X.reshape(-1, X.shape[-1]))
    DS_ = KroneckerExpanded(DS, C)
    
    @jit
    def F(theta):
        B = theta[:N * C].reshape(Y.shape)
        w = theta[N * C:].reshape(M, C)
        return B + (X @ w).reshape(Y.shape)
    
    @jit
    def mu(theta):
        F_ = jnp.exp(F(theta))
        return F_ / F_.sum(-1)[..., None]
    
    Muy = (mu(theta) - Y).reshape(-1)
    t1 = DS_ @ Muy
    t2 = jnp.kron(X.T @ DS, jnp.eye(C)) @ Muy
    
    return jnp.concatenate([t1, t2]) + KroneckerBlockDiag([gamma * KroneckerExpanded(U @ DGi2 @ U.T, C), lam * jnp.eye(M * C)]) @ theta


def test_all():

    X, Y, Y_logistic, Y_multiclass, Y_kgr, Y_kgr_logistic, Y_kgr_multiclass, S, S_kgr, U, G, gamma, lam, lamK, V = get_test_data(1)

    sol_gsr = solve_gsr(Y, S, U, G, gamma)
    sol_kgr = solve_kgr(Y_kgr, S_kgr, U, G, V, lamK, gamma)
    sol_rnc = solve_rnc(X, Y, S, U, G, gamma, lam)

    sol_lgsr = solve_lgsr(Y_logistic, S, U, G, gamma)
    sol_lkgr = solve_lkgr(Y_kgr_logistic, S_kgr, U, G, V, lamK, gamma)
    sol_lrnc = solve_lrnc(X, Y_logistic, S, U, G, gamma, lam)

    sol_lgsr_mc = solve_lgsr_multiclass(Y_multiclass, S, U, G, gamma)
    sol_lkgr_mc = solve_lkgr_multiclass(Y_kgr_multiclass, S_kgr, U, G, V, lamK, gamma)
    sol_lrnc_mc = solve_lrnc_multiclass(X, Y_multiclass, S, U, G, gamma, lam)


    grad_gsr_ = grad_gsr(sol_gsr, Y, S, U, G, gamma)
    grad_kgr_ = grad_kgr(sol_kgr, Y_kgr, S_kgr, U, G, V, lamK, gamma)
    grad_rnc_ = grad_rnc(sol_rnc, X, Y, S, U, G, gamma, lam)

    grad_lgsr_ = grad_lgsr(sol_lgsr, Y_logistic, S, U, G, gamma)
    grad_lkgr_ = grad_lkgr(sol_lkgr, Y_kgr_logistic, S_kgr, U, G, V, lamK, gamma)
    grad_lrnc_ = grad_lrnc(sol_lrnc, X, Y_logistic, S, U, G, gamma, lam)

    grad_lgsr_mc_ = grad_lgsr_multiclass(sol_lgsr_mc, Y_multiclass, S, U, G, gamma)
    grad_lkgr_mc_ = grad_lkgr_multiclass(sol_lkgr_mc, Y_kgr_multiclass, S_kgr, U, G, V, lamK, gamma)
    grad_lrnc_mc_ = grad_lrnc_multiclass(sol_lrnc_mc, X, Y_multiclass, S, U, G, gamma, lam)


    for grad, mod in zip([grad_gsr_, grad_kgr_, grad_rnc_, grad_lgsr_, grad_lkgr_, grad_lrnc_, grad_lgsr_mc_, grad_lkgr_mc_, grad_lrnc_mc_], 
                    ['gsr', 'kgr', 'rnc', 'lgsr', 'lkgr', 'lrnc', 'lgsr_mc', 'lkgr_mc', 'lrnc_mc']):
                
        assert np.allclose(grad ** 2, 0, atol=1e-5), f'Failed for {mod}: grad = {grad}, {grad ** 2 > 5e-5}'

    print('All checks passed')


if __name__ == '__main__':

    test_all()
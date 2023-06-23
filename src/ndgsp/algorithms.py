import numpy as np
import jax.numpy as jnp
from jax import jit
from pykronecker import KroneckerBlock, KroneckerBlockDiag, KroneckerProduct, KroneckerDiag, KroneckerIdentity

from ndgsp.lin_system import solve_SPCGM
from ndgsp.utils.arrays import outer_product
from ndgsp.operators import KroneckerExpanded, KroneckerMuBlock, KroneckerQXBlock


def assert_float64(*As):
    """
    Check that an input arrays have 64-bit floating point precision. This is necessary 
    for convergence. 
    """

    for A in As:

        if A.dtype != jnp.float64:
                
            err =  f"""All input arrays must be 64-bit floating point precision, but this array is {A.dtype}. To enable this, run \n\tfrom jax import config \n\tconfig.update("jax_enable_x64", True) \n at startup. See https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision"""

            raise ValueError(err)



@jit
def mu_logistic(F: jnp.array):
    return 1 / (1 + jnp.exp(-F))

@jit
def mu_softmax(F: jnp.array):
    F_ = jnp.exp(F)
    return F_ / F_.sum(-1)[..., None] 

@jit
def XRmuX(Mu: jnp.array, DSX: jnp.array):
    """
    Fast computation of (X ⊗ I).T Rmu (X ⊗ I) without using literal Kronecker product
    """

    M = DSX.shape[1]
    Mu = Mu.reshape(-1, Mu.shape[-1])
    out = [[0 for i in range(M)] for i in range(M)]

    for i in range(M):
                
        for j in range(i, M):

            X1 = DSX[:, i][:, None]
            X2 = DSX[:, j][:, None]
            X1_ = X1 * Mu
            X2_ = X2 * Mu
            out[i][j] = out[j][i] = jnp.diag((X1 * X2 * Mu).sum(0)) - jnp.asarray(X2_.T @ X1_)

    return jnp.block(out)    
    


def solve_gsr(Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float) -> np.ndarray:
    """
    Solve for posterior mean of Graph Signal Reconstruction model
    
    Params:
        Y:        ndarray (float) (N_1, ..., N_d) - partially observed input (with zeros for missing data)
        Y:        ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
    
    Returns:
        F:        ndarray (float) (N_1, ..., N_d) - the reconstructed graph signal
    """
    
    Y = jnp.asarray(Y)
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DG = KroneckerDiag(G)
    DS = KroneckerDiag(S)
    
    Phi = U @ DG
    Q = Phi.T @ DS @ Phi + gamma * KroneckerIdentity(like=DS)
    F, nits = solve_SPCGM(A_precon=Q, Y=Y, Phi=Phi)
    
    return np.asarray(F)


def sample_gsr(Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float, n_samples: int=1, seed: int=None) -> np.ndarray:
    """
    Sample from posterior of Graph Signal Reconstruction model
    
    Params:
        Y:           ndarray (float) (N_1, ..., N_d) - partially observed input (with zeros for missing data)
        Y:           ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:           KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:           ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:       float - graph regularisation parameter
        n_samples:   int - number of samples
        seed:        int - random seed
        
    Returns:
        Fs:        ndarray (float) (n_samples, N_1, ..., N_d) - samples of the reconstructed graph signal
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    Y = jnp.asarray(Y)
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)

    assert_float64(Y, S, G, U)

    DG = KroneckerDiag(G)
    DS = KroneckerDiag(S)
    
    Phi = U @ DG
    Q = Phi.T @ DS @ Phi + gamma * KroneckerIdentity(like=DS)
    
    Fs = []
    
    for i in range(n_samples):
        
        z1 = jnp.asarray(np.random.normal(size=Y.shape))
        z2 = jnp.asarray(np.random.normal(size=Y.shape))

        a1 = DG @ U.T @ DS @ z1
        a2 = gamma ** 0.5 * z2

        F, nits = solve_SPCGM(A_precon=Q, Y=(Y + a1 + a2), Phi=Phi)
        
        Fs.append(np.asarray(F))
    
    return Fs


def solve_kgr(Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, V: np.ndarray, lamK: np.ndarray, gamma: float) -> np.ndarray:
    """
    Solve for posterior mean of Kernel Graph Regression model
    
    Params:
        Y:        ndarray (float) (T, N_1, ..., N_d) - partially observed input (with zeros for missing data)
        Y:        ndarray (binary) (T, N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
        V:        ndarray (T, T) - eigenvector matrix of kernel matrix
        lamK:     ndarray (T, ) - eigenvalue vector of kernel matrix 
        
    Returns:
        F:        ndarray (float) (T, N_1, ..., N_d) - the reconstructed graph signal
    """
    
    V = jnp.asarray(V)
    lamK = jnp.asarray(lamK)
    
    if isinstance(U, KroneckerProduct):
        U_ = KroneckerProduct([V] + U.As)
    else:
        U_ = KroneckerProduct([V, U])

    G_ = outer_product(lamK ** 0.5, G)
    
    return solve_gsr(Y, S, U_, G_, gamma)


def sample_kgr(Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, V: np.ndarray, lamK: np.ndarray, gamma: float, n_samples: int=1, seed: int=None) -> np.ndarray:
    """
    Sample from posterior of Kernel Graph Regression model
    
    Params:
        Y:           ndarray (float) (T, N_1, ..., N_d) - partially observed input (with zeros for missing data)
        S:           ndarray (binary) (T, N_1, ..., N_d) - binary sensing tensor
        U:           KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:           ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:       float - graph regularisation parameter
        V:           ndarray (T, T) - eigenvector matrix of kernel matrix
        lamK:        ndarray (T, ) - eigenvalue vector of kernel matrix 
        n_samples:   int - number of samples
        seed:        int - random seed
        
    Returns:
        Fs:         ndarray (float) (n_samples, T, N_1, ..., N_d) - samples of the reconstructed graph signal
    """
        
    U_ = KroneckerProduct([jnp.asarray(V)] + U.As)
    G_ = outer_product(lamK ** 0.5, G)

    return sample_gsr(Y, S, U_, G_, gamma, n_samples, seed)
    
    
def solve_rnc(X: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """
    Solve for posterior mean of Regression with Network Cohesion model
    
    Params:
        X:        ndarray (float)  (N_1, ..., N_d, M) - tensor of explanatory variables
        Y:        ndarray (float) (N_1, ..., N_d) - partially observed input (with zeros for missing data)
        S:        ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
        lam:      float - feature regularisation parameter
    
    Returns:
        theta:    ndarray (float) (N + M, ) - the RNC parameter vector
    """
    
    Y = jnp.asarray(Y)
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DG = KroneckerDiag(G)
    DS = KroneckerDiag(S)
    X = jnp.asarray(X.reshape(-1, X.shape[-1]))
        
    lamM, UM = jnp.linalg.eigh(X.T @ DS @ X)
    DM = jnp.diag((lamM + lam) ** -0.5)
        
    Q11 = DG @ U.T @ DS @ U @ DG + gamma * KroneckerIdentity(like=DS)
    Q12 = DG @ U.T @ DS @ X @ UM @ DM
    Q21 = Q12.T
    Q22 = jnp.eye(X.shape[-1])
    
    Q = KroneckerBlock([[Q11, Q12], [Q21, Q22]])
    Phi = KroneckerBlockDiag([U @ DG, UM @ DM])
        
    y = Y.reshape(-1)
    Y_ = jnp.concatenate([y, X.T @ y])
        
    theta, nits = solve_SPCGM(A_precon=Q, Y=Y_, Phi=Phi)
        
    return np.asarray(theta)


def sample_rnc(X: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float, lam: float, n_samples: int=1, seed: int=None) -> np.ndarray:
    """
    Sample from posterior of Regression with Network Cohesion model
    
    Params:
        X:           ndaray (float)  (N_1, ..., N_d, M) - tensor of explanatory variables
        Y:           ndarray (float) (N_1, ..., N_d) - partially observed input (with zeros for missing data)
        Y:           ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:           KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:           ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:       float - graph regularisation parameter
        n_samples:   int - number of samples
        seed:        int - random seed
        
    ReturnsL
        thetas:      ndarray (float) (n_samples, N + M) - samples of the RNC parameter vector
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    Y = jnp.asarray(Y)
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DG = KroneckerDiag(G)
    DS = KroneckerDiag(S)
    X = jnp.asarray(X.reshape(-1, X.shape[-1]))        

    lamM, UM = jnp.linalg.eigh(X.T @ DS @ X)
    DM = jnp.diag((lamM + lam) ** -0.5)
        
    Q11 = DG @ U.T @ DS @ U @ DG + gamma * KroneckerIdentity(like=DS)
    Q12 = DG @ U.T @ DS @ X @ UM @ DM
    Q21 = Q12.T
    Q22 = jnp.eye(X.shape[-1])
    
    Q = KroneckerBlock([[Q11, Q12], [Q21, Q22]])
    Phi = KroneckerBlockDiag([U @ DG, UM @ DM])
    
    y = Y.reshape(-1)
    Y_ = jnp.concatenate([y, X.T @ y])
    
    N = np.prod(Y.shape)
    M = X.shape[-1]
    
    A1 = KroneckerBlock([[DG @ U.T @ DS, np.zeros((N, M))], [DM @ UM.T @ X.T @ DS, np.zeros((M, M))]])
    A2 = KroneckerBlockDiag([gamma ** 0.5 * KroneckerIdentity(like=DG), lam ** 0.5 * jnp.eye(M)])
    
    thetas = []
    
    for i in range(n_samples):
        
        z1 = np.random.normal(size=N + M)
        z2 = np.random.normal(size=N + M)
        
        a1 = A1 @ z1
        a2 = A2 @ z2

        theta, nits = solve_SPCGM(A_precon=Q, Y=(Y_ + a1 + a2), Phi=Phi)

        thetas.append(theta)
        
    return np.squeeze(np.asarray(thetas))


def solve_lgsr(Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float) -> np.ndarray:
    """
    Solve for posterior mean of Logistic Graph Signal Reconstruction Model
    
    Params:
        Y:        ndarray (binary) (N_1, ..., N_d) - partially observed input (with zeros for missing data)
        S:        ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
        
    Returns:
        F:        ndarray (float) (N_1, ..., N_d) - the real valued latent graph signal (untransformed)
    """
    
    Y = jnp.asarray(Y.astype(float))
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)

    assert_float64(Y, S, U, G)

    DG = KroneckerDiag(G)
    DS = KroneckerDiag(S)
    Phi = U @ DG
    
    F = solve_gsr(2 * Y - S, S, U, G, gamma)
    dF = F.copy()
    
    N = np.prod(Y.shape)
    
    while np.abs(dF).sum() / N > 1e-6:
        
        M = mu_logistic(F)
        Dmu = KroneckerDiag(S * M * (1 - M))
        T = DS @ (Y - M) + Dmu @ F

        Q = Phi.T @ Dmu @ Phi + gamma * KroneckerIdentity(like=DS)
        F_, nits = solve_SPCGM(A_precon=Q, Y=T, Phi=Phi)
        dF = F - F_
        F = F_
        
    return np.asarray(F)


def solve_lkgr(Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, V: np.ndarray, lamK: np.ndarray, gamma: float) -> np.ndarray:
    """
    Solve for posterior mean of Logistic Kernel Graph Regression model
    
    Params:
        Y:        ndarray (binary) (T, N_1, ..., N_d) - partially observed input (with zeros for missing data)
        S:        ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
        V:        ndarray (T, T) - eigenvector matrix of kernel matrix
        lamK:     ndarray (T, ) - eigenvalue vector of kernel matrix 
        
    Returns:
        F:        ndarray (float) (T, N_1, ..., N_d) - the real valued latent graph signal (untransformed)
    """
    
    V = jnp.asarray(V)
    lamK = jnp.asarray(lamK)
    
    if isinstance(U, KroneckerProduct):
        U_ = KroneckerProduct([V] + U.As)
    else:
        U_ = KroneckerProduct([V, U])
        
    G_ = outer_product(lamK ** 0.5, G)
    
    return solve_lgsr(Y, S, U_, G_, gamma)


def solve_lrnc(X: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """
    Solve for posterior mean of Logistic Regression with Network Cohesion model
    
    Params:
        X:        ndaray (float)  (N_1, ..., N_d, M) - tensor of explanatory variables
        Y:        ndarray (float) (N_1, ..., N_d) - partially observed input (with zeros for missing data)
        Y:        ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
        lam:      float - feature regularisation parameter
        
    Returns:
        theta:    ndarray (float) (N + M, ) - the RNC parameter vector (untransformed)
    """
    
    Y = jnp.asarray(Y.astype(float))
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DG = KroneckerDiag(G)
    DS = KroneckerDiag(S)
    X = jnp.asarray(X.reshape(-1, X.shape[-1]))

    assert_float64(X, Y, S, U, G)
    
    N = np.prod(Y.shape)
    M = X.shape[-1]
    theta = solve_rnc(X, 2 * Y - S, S, U, G, gamma, lam)
    dtheta = theta.copy()

    i = 0
            
    while np.abs(dtheta).sum() / (N + M) > 1e-6:

        
        F = (theta[:N] + X @ theta[N:]).reshape(Y.shape)
        Mu = mu_logistic(F)
        Dmu = KroneckerDiag(S * Mu * (1 - Mu))
        lamM, UM = jnp.linalg.eigh(X.T @ Dmu @ X)
        DM = jnp.diag((lamM + lam) ** -0.5)
        
        Q11 = DG @ U.T @ Dmu @ U @ DG + gamma * KroneckerIdentity(like=DS)
        Q12 = DG @ U.T @ Dmu @ X @ UM @ DM
        Q21 = Q12.T
        Q22 = jnp.eye(X.shape[-1])

        Q = KroneckerBlock([[Q11, Q12], [Q21, Q22]])
        Phi = KroneckerBlockDiag([U @ DG, UM @ DM])
        
        tt = (Y - Mu + Dmu @ F).reshape(-1)
        t = jnp.concatenate([DS @ tt, X.T @ DS @ tt])
        
        theta_, nits = solve_SPCGM(A_precon=Q, Y=t, Phi=Phi)
        dtheta = theta - theta_
        theta = theta_
        
        i += 1

        if i > 20:
            break

    return np.asarray(theta)


def solve_lgsr_multiclass(Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float) -> np.ndarray:
    """
    Solve for posterior mean of Multiclass Logistic Graph Signal Reconstruction Model
    
    Params:
        Y:        ndarray (binary) (N_1, ..., N_d, C) - partially observed one-hot encoded input (with zeros for missing data)
        Y:        ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
        
    Returns:
        F:        ndarray (float) (N_1, ..., N_d, C) - the real valued latent graph signal (untransformed)
    """
    
    C = Y.shape[-1]
    N = np.prod(Y.shape)

    Y = jnp.asarray(Y.astype(float))
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DG = KroneckerDiag(G)
    DS = KroneckerDiag(S)
    
    Phi = KroneckerExpanded(U @ DG, C)
    DS_ = KroneckerExpanded(DS, C)

    F, _ = solve_SPCGM(A_precon=Phi.T @ DS_ @ Phi + gamma * KroneckerIdentity(like=DS_), Y=2 * Y - S[..., None], Phi=Phi)
    dF = F.copy()
    
    while np.abs(dF).sum() / N > 1e-6:
        
        Mu = mu_softmax(F)
        Rmu = DS_ @ (KroneckerDiag(Mu) - KroneckerMuBlock(Mu))
        T = DS_ @ (Y - Mu) + Rmu @ F
        Q = Phi.T @ Rmu @ Phi + gamma * KroneckerIdentity(like=Rmu)
        F_, nits = solve_SPCGM(A_precon=Q, Y=T, Phi=Phi)
        dF = F - F_
        F = F_
        
    return np.asarray(F)


def solve_lkgr_multiclass(Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, V: np.ndarray, lamK: np.ndarray, gamma: float) -> np.ndarray:
    """
    Solve for posterior mean of Multiclass Logistic Kernel Graph Regression model
    
    Params:
        Y:        ndarray (binary) (T, N_1, ..., N_d, C) - partially observed one-hot encoded input (with zeros for missing data)
        S:        ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
        V:        ndarray (T, T) - eigenvector matrix of kernel matrix
        lamK:     ndarray (T, ) - eigenvalue vector of kernel matrix 
        
    Returns:
        F:        ndarray (float) (T, N_1, ..., N_d, C) - the real valued latent graph signal (untransformed)
    """
    
    V = jnp.asarray(V)
    lamK = jnp.asarray(lamK)
    
    if isinstance(U, KroneckerProduct):
        U_ = KroneckerProduct([V] + U.As)
    else:
        U_ = KroneckerProduct([V, U])

    G_ = outer_product(lamK ** 0.5, G)
    
    return solve_lgsr_multiclass(Y, S, U_, G_, gamma)



def solve_lrnc_multiclass(X: np.ndarray, Y: np.ndarray, S: np.ndarray, U: KroneckerProduct, G: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """
    Solve for posterior mean of Logistic Regression with Network Cohesion model
    
    Params:
        X:        ndaray (float)  (N_1, ..., N_d, M) - tensor of explanatory variables
        Y:        ndarray (binary) (N_1, ..., N_d, C) - partially observed one-hot encoded input (with zeros for missing data)
        Y:        ndarray (binary) (N_1, ..., N_d) - binary sensing tensor
        U:        KroneckerProduct (float) (N, N) - GFT eigenvector matrix
        G:        ndarray (float) (N_1, ..., N_d) - spectral scaling tensor
        gamma:    float - graph regularisation parameter
        lam:      float - feature regularisation parameter
        
    Returns:
        theta:    ndarray (float) (NC + MC, )  - the RNC parameter vector (untransformed)
    """
    
    N = np.prod(G.shape)
    M = X.shape[-1]
    C = Y.shape[-1]
    
    Y = jnp.asarray(Y.astype(float))
    S = jnp.asarray(S.astype(float))
    G = jnp.asarray(G)
    DG = KroneckerDiag(G)
    DS = KroneckerDiag(S)
    X = jnp.asarray(X.reshape(-1, X.shape[-1]))
    DS_ = KroneckerExpanded(DS, C)

    Phi1 = KroneckerExpanded(U @ DG, C)

    thetas = [solve_rnc(X, (2 * Y[..., c] - S), S, U, G, gamma, lam) for c in range(C)]

    Cs = jnp.stack([thetas[c][:N] for c in range(C)], axis=-1)
    Ws = jnp.stack([thetas[c][N:] for c in range(C)], axis=-1)

    theta = jnp.concatenate([Cs.ravel(), Ws.ravel()])

    dtheta = theta.copy()

    while np.abs(dtheta).sum() / (N * C + M * C) > 1e-6:

        print(np.abs(dtheta).sum() / (N * C + M * C))

        F = theta[:N * C].reshape(Y.shape) + (X @ theta[N * C:].reshape(M, C)).reshape(Y.shape)    
        Mu = mu_softmax(F)
        Rmu = DS_ @ (KroneckerDiag(Mu) - KroneckerMuBlock(Mu))
        
        lamM, UM = jnp.linalg.eigh(XRmuX(Mu, DS @ X))
        DM = jnp.diag((lamM + lam) ** -0.5)
        Phi2 = UM @ DM
        
        tt = (Y - Mu + Rmu @ F).reshape(-1)
        
        t1 = DS_ @ tt
        t2 = (X.T @ DS @ tt.reshape(-1, C)).ravel()
        t = jnp.concatenate([t1, t2])
        
        Q11 = Phi1.T @ Rmu @ Phi1 + gamma * KroneckerIdentity(like=DS_)
        Q12 = KroneckerQXBlock(Phi1.T @ Rmu, X, Phi2)
        Q21 = Q12.T
        Q22 = jnp.eye(M * C)

        Q = KroneckerBlock([[Q11, Q12], [Q21, Q22]])
        Phi = KroneckerBlockDiag([Phi1, Phi2])

        theta_, nits = solve_SPCGM(A_precon=Q, Y=t, Phi=Phi)
        dtheta = theta - theta_
        theta = theta_
        
    return np.asarray(theta)



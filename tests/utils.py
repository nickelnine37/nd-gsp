import numpy as np
import jax.numpy as jnp
import networkx as nx


def get_random_A(N: int, array_type: str='numpy') -> np.ndarray | jnp.ndarray:
    """
    Get a random valid adjacency matrix. array_type should be numpy or jax
    """
    A = (np.random.randint(0, 2, size=(N, N)) * (1 - np.eye(N))).astype(bool)
    A = (A + A.T).astype(float)

    if array_type == 'numpy':
        return A

    elif array_type == 'jax':
        return jnp.asarray(A)


def get_random_L(N: int, array_type: str='numpy') -> np.ndarray | jnp.ndarray:
    """
    Get a random valid Laplacian matrix. array_type should be numpy or jax
    """

    A = get_random_A(N, array_type='numpy')
    L = np.diag(A.sum(0)) - A

    if array_type == 'numpy':
        return L

    elif array_type == 'jax':
        return jnp.asarray(L)


def get_random_nx(N: int):
    """
    Get a random networx graph
    """

    A = get_random_A(N, array_type='numpy')

    return nx.from_numpy_array(A)



import numpy as np
from numpy import ndarray, diag
from scipy.sparse import csr_array, spmatrix, triu as sptriu, tril as sptril
from typing import Union



def check_valid_adjacency(A: Union[ndarray, spmatrix], directed=False, self_loops=False) -> bool:
    """
    Perform some basic checks on the adjacency matrix. Returns true if no problems are
    found, else raises an assertion error

    Parameters
    ----------
    A               Adjacency matrix
    directed        Whether graph is directed
    self_loops      Whether self-loops are allowed

    """

    assert isinstance(A, Union[ndarray, spmatrix]), 'A must be an ndarray or an spmatrix'

    assert A.ndim == 2, 'A must be 2-dimensional'
    assert A.shape[0] == A.shape[1], 'A must be square'

    if not directed:
        assert (A.T != A).sum() == 0, 'A is not symmetric'

    if not self_loops:
        assert not A.diagonal().any(), 'A has non-zero entries along the diagonal'

    return True


def check_valid_laplacian(L: Union[ndarray, spmatrix]) -> bool:
    """
    Perform some basic checks on the Laplacian matrix. Returns true if no problems are
    found, else raises an assertion error. Directed graphs are not permitted here.

    Parameters
    ----------
    L               Laplacian matrix
    """

    assert isinstance(L, Union[ndarray, spmatrix]), 'L must be an ndarray or an spmatrix'

    assert L.ndim == 2, 'L must be 2-dimensional'
    assert L.shape[0] == L.shape[1], 'L must be square'

    if isinstance(L, ndarray):
        assert ((-np.triu(L, 1) - np.tril(L, -1)).sum(0) == L.diagonal()).all(), 'Laplacian matrix does not appear to have come from a valid adjacency matrix'

    if isinstance(L, spmatrix):
        assert (csr_array((-sptriu(L, 1) - sptril(L, -1))).sum(1) == L.diagonal()).all(), 'Laplacian matrix does not appear to have come from a valid adjacency matrix'

    return True



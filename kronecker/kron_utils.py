import numpy as np
from numpy import ndarray
from scipy.sparse import spmatrix

from kronecker.kron_base import KroneckerOperator


def check_valid_matrices(*As) -> bool:

    assert all(isinstance(A, (ndarray, spmatrix)) for A in As)
    assert all(A.ndim == 2 for A in As)
    assert all(A.shape[0] == A.shape[1] for A in As)

    return True

def check_operators_consistent(*operators) -> bool:

    assert all(isinstance(A, KroneckerOperator) for A in operators), f'All operators in this chain must be consistent, but they have types {[type(operator) for operator in operators]} respectively'
    assert all(op1.shape == op2.shape for op1, op2 in zip(operators[1:], operators[:-1])), f'All operators in this chain should have the same shape, but they have shapes {[operator.shape for operator in operators]} respectively'

    return True

def check_blocks_consistent(blocks: list):

    ndim = np.asarray(blocks, dtype='object').ndim

    if ndim == 1:
        assert all(isinstance(block, (KroneckerOperator, ndarray, spmatrix)) for block in blocks)
        assert all(block.shape[0] == block.shape[1] for block in blocks)

    elif ndim == 2:

        # check diagonal blocks are square
        assert all(blocks[i][i].shape[0] == blocks[i][i].shape[1] for i in range(len(blocks)))
        shapes = [blocks[i][i].shape[0] for i in range(len(blocks))]

        for i in range(len(blocks)):
            for j in range(len(blocks)):
                assert isinstance(blocks[i][j], (KroneckerOperator, ndarray, spmatrix))
                assert blocks[i][j].shape == (shapes[i], shapes[j])

    else:
        raise ValueError(f'blocks should be 1d or 2d but it is {np.ndim(blocks)}d')

    return True

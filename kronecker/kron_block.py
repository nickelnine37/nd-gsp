import numpy as np
from numpy import ndarray
from kronecker.kron_base import KroneckerOperator
from kronecker.kron_utils import check_blocks_consistent


class KroneckerBlock(KroneckerOperator):

    def __init__(self, blocks: list):
        """
        Create a general block operator. Items in the block can be arrays or operators.

        E.g. blocks = [[A11, A12, A13]
                       [A21, A22, A23]
                       [A31, A32, A33]]
        """

        check_blocks_consistent(blocks)
        self.blocks = blocks
        self.n_blocks = len(self.blocks)
        self.block_sizes = [self.blocks[i][i].shape[0] for i in range(self.n_blocks)]
        self.cum_block_sizes = [0] + np.cumsum(self.block_sizes).tolist()
        self.iter_edges = lambda: zip(self.cum_block_sizes[:-1], self.cum_block_sizes[1:])

        N = sum(self.block_sizes)
        self.shape = (N, N)

    def __copy__(self) -> 'KroneckerBlock':
        new = KroneckerBlock(blocks=[[self.blocks[i][j].copy() if isinstance(self.blocks[i][j] , KroneckerOperator) else self.blocks[i][j] for i in range(self.n_blocks)] for j in range(self.n_blocks)])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}) -> 'KroneckerBlock':
        new = KroneckerBlock(blocks=[[self.blocks[i][j].deepcopy() if isinstance(self.blocks[i][j] , KroneckerOperator) else self.blocks[i][j].copy() for i in range(self.n_blocks)] for j in range(self.n_blocks)])
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        new = KroneckerBlock(blocks=[[self.blocks[i][j] ** power for i in range(self.n_blocks)] for j in range(self.n_blocks)])
        new.factor = self.factor ** power
        return new


    def operate(self, other: ndarray) -> ndarray:


        if other.ndim == 1:

            assert len(other) == self.shape[1]

            out = [np.zeros_like(other[n1:n2]) for n1, n2 in self.iter_edges()]
            other = [other[n1:n2] for n1, n2 in self.iter_edges()]

        elif other.ndim == 2:

            assert other.shape[0] == self.shape[1]

            out = [np.zeros_like(other[n1:n2, :]) for n1, n2 in self.iter_edges()]
            other = [other[n1:n2, :] for n1, n2 in self.iter_edges()]

        else:
            raise ValueError(f'other must be 1 or 2d but it is {other.ndim}d')

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                out[i] += self.blocks[i][j] @ other[j]

        return self.factor * np.concatenate(out, axis=0)


    def inv(self):
        raise NotImplementedError

    @property
    def T(self):
        return self.factor * KroneckerBlock(blocks=[[self.blocks[j][i].T for i in range(self.n_blocks)] for j in range(self.n_blocks)])

    def conj(self):
        return self.factor * KroneckerBlock(blocks=[[self.blocks[j][i].conj() for i in range(self.n_blocks)] for j in range(self.n_blocks)])

    def to_array(self) -> ndarray:
        return self.factor * np.block([[self.blocks[j][i].to_array() if isinstance(self.blocks[j][i], KroneckerOperator) else self.blocks[j][i] for i in range(self.n_blocks)] for j in range(self.n_blocks)])

    def __repr__(self):

        def to_string(block):
            return str(block) if isinstance(block, KroneckerOperator) else f'ndarray({block.shape})'

        return 'KroneckerBlock([{}])'.format(', '.join(['[' + ', '.join([to_string(self.blocks[i][j]) for j in range(self.n_blocks)]) + ']' for i in range(self.n_blocks)]))

    def __str__(self):
        return self.__repr__()


class KroneckerBlockDiag(KroneckerOperator):

    def __init__(self, blocks: list):
        """
        Create a diagonal block operator. Items in the block can be arrays or operators.

        E.g. blocks = [A1, A2, A3] -> [[A1, 0, 0]
                                       [0, A2, 0]
                                       [0, 0, A3]]
        """

        check_blocks_consistent(blocks)
        self.blocks = blocks
        self.n_blocks = len(self.blocks)
        self.block_sizes = [block.shape[0] for block in self.blocks]
        self.cum_block_sizes = [0] + np.cumsum(self.block_sizes).tolist()
        self.iter_edges = lambda: zip(self.cum_block_sizes[:-1], self.cum_block_sizes[1:])

        N = sum(self.block_sizes)
        self.shape = (N, N)

    def __copy__(self) -> 'KroneckerOperator':
        new = KroneckerBlockDiag(blocks=[block.copy() if isinstance(block, KroneckerOperator) else block for block in self.blocks])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}) -> 'KroneckerOperator':
        new = KroneckerBlockDiag(blocks=[block.deepcopy() if isinstance(block, KroneckerOperator) else block.copy() for block in self.blocks])
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        return self.factor ** power * KroneckerBlockDiag(blocks=[block ** power  for block in self.blocks])

    def operate(self, other: ndarray) -> ndarray:
        """
        other should be a vector only
        """

        if other.ndim == 1:
            assert len(other) == self.shape[1]
            return self.factor * np.concatenate([block @ other[n1:n2] for block, (n1, n2) in zip(self.blocks, self.iter_edges())], axis=0)

        elif other.ndim == 2:
            assert other.shape[0] == self.shape[1]
            return self.factor * np.concatenate([block @ other[n1:n2, :] for block, (n1, n2) in zip(self.blocks, self.iter_edges())], axis=0)

        else:
            raise ValueError('other must be 1 or 2d')


    def inv(self) -> 'KroneckerBlockDiag':
        return self.factor ** -1 * KroneckerBlockDiag(blocks=[block.inv() for block in self.blocks])

    @property
    def T(self):
        return self.factor * KroneckerBlockDiag(blocks=[block.T for block in self.blocks])

    def conj(self):
        return self.factor * KroneckerBlockDiag(blocks=[block.conj() for block in self.blocks])

    def to_array(self) -> ndarray:

        out = np.zeros(self.shape)

        for block, (n1, n2) in zip(self.blocks, self.iter_edges()):

            if isinstance(block, KroneckerOperator):
                out[n1:n2, n1:n2] = block.to_array()
            else:
                out[n1:n2, n1:n2] = block

        return out

    def __repr__(self):
        return 'KroneckerBlockDiag([{}])'.format(', '.join([str(block) if isinstance(block, KroneckerOperator) else f'ndarray{block.shape}' for block in self.blocks]))

    def __str__(self):
        return 'KroneckerBlockDiag([{}])'.format(', '.join([str(block) if isinstance(block, KroneckerOperator) else f'ndarray{block.shape}' for block in self.blocks]))

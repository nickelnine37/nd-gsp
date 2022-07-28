from typing import Callable

import numpy as np
from numpy import ndarray
import kronecker.kron_base


class KroneckerBlock(kronecker.kron_base.KroneckerOperator):

    def __init__(self, blocks: list[list]):
        """
        Create a general block operator. Items in the block can be arrays or operators.

        E.g. blocks = [[A11, A12, A13]
                       [A21, A22, A23]
                       [A31, A32, A33]]
        """

        kronecker.kron_base.check_blocks_consistent(blocks)
        
        self.state = {'blocks': blocks, 
                      'block_sizes': [blocks[i][i].shape[0] for i in range(len(blocks))], 
                      'n_blocks': len(blocks)}

        self.state.update({'cum_block_sizes': [0] + np.cumsum(self.state['block_sizes']).tolist()})
        
        self.state.update({'iter_edges': lambda: zip(self.state['cum_block_sizes'][:-1], self.state['cum_block_sizes'][1:])})
        
        N = sum(self.state['block_sizes'])
        self.shape = (N, N)
        
    def apply_to_blocks(self, function: Callable, transpose=False):
        """
        Helper method: apply `function` to each block, and return as nested list
        """
    
        if transpose:
            return [[function(self.state['blocks'][j][i]) for j in range(self.state['n_blocks'])] for i in range(self.state['n_blocks'])]
        else:
            return [[function(self.state['blocks'][i][j]) for j in range(self.state['n_blocks'])] for i in range(self.state['n_blocks'])]

    def __copy__(self) -> 'KroneckerBlock':
        new = KroneckerBlock(blocks=self.apply_to_blocks(lambda block: block.copy() if isinstance(block, kronecker.kron_base.KroneckerOperator) else block))
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}) -> 'KroneckerBlock':
        new = KroneckerBlock(blocks=self.apply_to_blocks(lambda block: block.deepcopy() if isinstance(block, kronecker.kron_base.KroneckerOperator) else block.copy()))
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        new = KroneckerBlock(blocks=self.apply_to_blocks(lambda block: block ** power))
        new.factor = self.factor ** power
        return new

    def operate(self, other: ndarray) -> ndarray:

        if other.ndim == 1:

            assert len(other) == self.shape[1]

            out = [np.zeros_like(other[n1:n2]) for n1, n2 in self.state['iter_edges']()]
            other = [other[n1:n2] for n1, n2 in self.state['iter_edges']()]

        elif other.ndim == 2:

            assert other.shape[0] == self.shape[1]

            out = [np.zeros_like(other[n1:n2, :]) for n1, n2 in self.state['iter_edges']()]
            other = [other[n1:n2, :] for n1, n2 in self.state['iter_edges']()]

        else:
            raise ValueError(f'other must be 1 or 2d but it is {other.ndim}d')

        for i in range(self.state['n_blocks']):
            for j in range(self.state['n_blocks']):
                out[i] += self.state['blocks'][i][j] @ other[j]

        return self.factor * np.concatenate(out, axis=0)

    def inv(self) -> 'KroneckerBlock':
        raise NotImplementedError

    @property
    def T(self) -> 'KroneckerBlock':
        return self.factor * KroneckerBlock(blocks=self.apply_to_blocks(lambda block: block.T, transpose=True))

    def conj(self) -> 'KroneckerBlock':
        return self.factor * KroneckerBlock(blocks=self.apply_to_blocks(lambda block: block.conj(), transpose=True))

    def to_array(self) -> ndarray:
        return self.factor * np.block(self.apply_to_blocks(lambda block: block.to_array() if isinstance(block, kronecker.kron_base.KroneckerOperator) else block))

    def __repr__(self) -> str:

        def to_string(block):
            return str(block) if isinstance(block, kronecker.kron_base.KroneckerOperator) else f'ndarray({block.shape})'

        return 'KroneckerBlock([{}])'.format(', '.join(['[' + ', '.join([to_string(self.state['blocks'][i][j]) for j in range(self.state['n_blocks'])]) + ']' for i in range(self.state['n_blocks'])]))



class KroneckerBlockDiag(kronecker.kron_base.KroneckerOperator):

    def __init__(self, blocks: list):
        """
        Create a diagonal block operator. Items in the block can be arrays or operators.

        E.g. blocks = [A1, A2, A3] -> [[A1, 0, 0]
                                       [0, A2, 0]
                                       [0, 0, A3]]
        """

        kronecker.kron_base.check_blocks_consistent(blocks)

        self.state = {'blocks': blocks,
                      'block_sizes': [blocks[i].shape[0] for i in range(len(blocks))],
                      'n_blocks': len(blocks)}

        self.state.update({'cum_block_sizes': [0] + np.cumsum(self.state['block_sizes']).tolist()})

        self.state.update({'iter_edges': lambda: zip(self.state['cum_block_sizes'][:-1], self.state['cum_block_sizes'][1:])})

        N = sum(self.state['block_sizes'])
        self.shape = (N, N)
        
        
    def apply_to_blocks(self, function: Callable):
        return [function(self.state['blocks'][i]) for i in range(self.state['n_blocks'])]
        
    
    def __copy__(self) -> 'kronecker.kron_base.KroneckerOperator':
        new = KroneckerBlockDiag(blocks=self.apply_to_blocks(lambda block: block.copy() if isinstance(block, kronecker.kron_base.KroneckerOperator) else block))
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}) -> 'kronecker.kron_base.KroneckerOperator':
        new = KroneckerBlockDiag(blocks=self.apply_to_blocks(lambda block: block.deepcopy() if isinstance(block, kronecker.kron_base.KroneckerOperator) else block.copy()))
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        new = KroneckerBlockDiag(blocks=self.apply_to_blocks(lambda block: block ** power))
        new.factor = self.factor ** power
        return new

    def operate(self, other: ndarray) -> ndarray:
        """
        other should be a vector only
        """

        if other.ndim == 1:
            assert len(other) == self.shape[1]
            return self.factor * np.concatenate([block @ other[n1:n2] for block, (n1, n2) in zip(self.state['blocks'], self.state['iter_edges']())], axis=0)

        elif other.ndim == 2:
            assert other.shape[0] == self.shape[1]
            return self.factor * np.concatenate([block @ other[n1:n2, :] for block, (n1, n2) in zip(self.state['blocks'], self.state['iter_edges']())], axis=0)

        else:
            raise ValueError('other must be 1 or 2d')


    def inv(self) -> 'KroneckerBlockDiag':
        return self.factor ** -1 * KroneckerBlockDiag(blocks=[block.inv() for block in self.state['blocks']])

    @property
    def T(self) -> 'KroneckerBlockDiag':
        return self.factor * KroneckerBlockDiag(blocks=self.apply_to_blocks(lambda block: block.T))

    def conj(self) -> 'KroneckerBlockDiag':
        return self.factor * KroneckerBlockDiag(blocks=self.apply_to_blocks(lambda block: block.conj()))

    def to_array(self) -> ndarray:

        out = np.zeros(self.shape)

        for block, (n1, n2) in zip(self.state['blocks'], self.state['iter_edges']()):

            if isinstance(block, kronecker.kron_base.KroneckerOperator):
                out[n1:n2, n1:n2] = block.to_array()
            else:
                out[n1:n2, n1:n2] = block

        return out

    def __repr__(self):
        return 'KroneckerBlockDiag([{}])'.format(', '.join([str(block) if isinstance(block, kronecker.kron_base.KroneckerOperator) else f'ndarray{block.shape}' for block in self.state['blocks']]))

    def __str__(self):
        return 'KroneckerBlockDiag([{}])'.format(', '.join([str(block) if isinstance(block, kronecker.kron_base.KroneckerOperator) else f'ndarray{block.shape}' for block in self.state['blocks']]))



if __name__ == '__main__':

    from utils.linalg import vec

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised = kronecker.kron_base.generate_test_data()
    x = np.concatenate([vec(X), vec(Y)])
    Q = np.concatenate([P, P], axis=0)

    kb_literal = np.block([[kp_literal, kd_literal], [np.zeros(kp_literal.shape), ks_literal]])
    kbd_literal = np.block([[kp_literal, np.zeros(kp_literal.shape)], [np.zeros(kp_literal.shape), ks_literal]])

    kb_optimised = KroneckerBlock([[kp_optimised, kd_optimised], [np.zeros(kp_literal.shape), ks_optimised]])
    kbd_optimised = KroneckerBlockDiag([kp_optimised, ks_optimised])

    kronecker.kron_base.run_assertions(x, Q, kb_literal, kb_optimised)
    kronecker.kron_base.run_assertions(x, Q, kbd_literal, kbd_optimised)

    print('kron_block.py: All tests passed')

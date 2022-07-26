import numpy as np
from numpy import ndarray
from kronecker.kron_base import KroneckerOperator
from kronecker.kron_utils import check_operators_consistent

"""
We use the classes in this file to create composite operators. This is the result of adding or multiplying 
two simpler operators together. These classes never need to be created explicityly, but are implicitly 
created whenever two operators are summed or multiplied. 

E.g.

>>> A = KroneckerProduct(A1, A2, A3)
>>> B = KroneckerSum(B1, B2, B3)

>>> C1 = A + B
>>> assert isinstance(C1, _SumChain)

>>> C2 = A @ B
>>> assert isinstance(C1, _ProductChain)

This abstraction can be used indefinitely to create higher and higher order composite operators. 
"""



class _SumChain(KroneckerOperator):
    """
    Used to represent a chain of Kronecker objects summed together
    """

    def __init__(self, *operators):
        check_operators_consistent(*operators)
        self.chain = operators
        self.shape = self.chain[0].shape
        self.shapes = self.chain[0].shapes

    def __copy__(self):
        new = _SumChain(*[operator.__copy__() for operator in self.chain])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = _SumChain(*[operator.__deepcopy__() for operator in self.chain])
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * sum(operator.operate(other) for operator in self.chain)

    def inv(self):
        raise NotImplementedError

    @property
    def T(self):
        return self.factor * _SumChain(*[operator.T for operator in self.chain])

    def conj(self):
        return self.factor * _SumChain(*[operator.conj() for operator in self.chain])

    def to_array(self) -> ndarray:
        return self.factor * sum(operator.to_array() for operator in self.chain)

    # def __repr__(self):
    #     return 'SumChain({})'.format(', '.join([str(operator) for operator in self.chain]))
    #
    # def __str__(self):
    #     return 'SumChain({})'.format(', '.join([str(operator) for operator in self.chain]))


class _ProductChain(KroneckerOperator):
    """
    Used to represent a chain of Kronecker objects matrix-multiplied together
    """

    def __init__(self, *operators):
        check_operators_consistent(*operators)
        self.chain = operators
        self.shape = self.chain[0].shape
        self.shapes = self.chain[0].shapes

    def __copy__(self):
        new = _ProductChain(*[operator.__copy__() for operator in self.chain])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = _ProductChain(*[operator.__deepcopy__() for operator in self.chain])
        new.factor = self.factor
        return new

    def operate(self, other: ndarray) -> ndarray:

        out = self.chain[-1] @ other
        for operator in reversed(self.chain[:-1]):
            out = operator @ out

        return self.factor * out

    def inv(self):
        return self.factor * _ProductChain(*[operator.inv() for operator in reversed(self.chain)])

    @property
    def T(self):
        return self.factor * _ProductChain(*[operator.T for operator in reversed(self.chain)])

    def conj(self):
        return self.factor * _ProductChain(*[operator.conj() for operator in reversed(self.chain)])

    def to_array(self) -> ndarray:
        out = self.chain[-1].to_array()
        for A in reversed(self.chain[:-1]):
            out = A.to_array() @ out
        return self.factor * out

    # def __repr__(self):
    #     return 'ProductChain({})'.format(', '.join([str(operator) for operator in self.chain]))
    #
    # def __str__(self):
    #     return 'ProductChain({})'.format(', '.join([str(operator) for operator in self.chain]))

    def __pow__(self, power, modulo=None):
        raise NotImplementedError



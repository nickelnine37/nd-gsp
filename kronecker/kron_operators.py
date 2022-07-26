import numpy as np
from numpy import ndarray
from typing import Union
from numpy.linalg import inv

from kronecker.kron_base import KroneckerOperator
from kronecker.kron_utils import check_valid_matrices, check_operators_consistent, check_blocks_consistent
from utils.linalg import vec, multiply_tensor_product, multiply_tensor_sum, ten, kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal
from scipy.sparse import spmatrix
from numbers import Number


class KroneckerProduct(KroneckerOperator):
    """
    Used to represent the object (A1 ⊗ A2 ⊗ ... ⊗ AN), that is the Kronecker product of N square matrices.
    """

    def __init__(self, *As):
        check_valid_matrices(*As)
        self.As = As
        self.ndim = len(As)
        self.shapes = tuple(A.shape[0] for A in As)
        N = int(np.prod(self.shapes))
        self.shape = (N, N)

    def __copy__(self):
        new = KroneckerProduct(*[A for A in self.As])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = KroneckerProduct(*[A.copy() for A in self.As])
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        return self.factor ** power * KroneckerProduct(*[A ** power for A in self.As])

    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        # in this case, if other is another Kronecker product, we can get a simpler representation
        if isinstance(other, KroneckerProduct):
            check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerProduct(*[A1 @ A2 for A1, A2 in zip(self.As, other.As)])

        else:
            return super().__matmul__(other)

    def __mul__(self, other):

        # kronecker products can be hadamarded against other kronecker products only
        if isinstance(other, KroneckerProduct):
            check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerProduct(*[A1 * A2 for A1, A2 in zip(self.As, other.As)])
        else:
            return super().__mul__(other)

    def operate(self, other: ndarray) -> ndarray:

        # handle when other is a vector
        if other.ndim == 1:
            return self.factor * vec(multiply_tensor_product(ten(other, shape=tuple(reversed(self.shapes))), *self.As))

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == self.shape[1]:
            out = np.zeros_like(other)
            for i in range(other.shape[1]):
                out[:, i] = vec(multiply_tensor_product(ten(other[:, i], shape=tuple(reversed(self.shapes))), *self.As))
            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * multiply_tensor_product(other, *self.As)

    def inv(self):
        return self.factor * KroneckerProduct(*[inv(A) for A in self.As])

    @property
    def T(self):
        return self.factor * KroneckerProduct(*[A.T for A in self.As])

    def conj(self):
        return self.factor * KroneckerProduct(*[A.conj() for A in self.As])

    def to_array(self) -> ndarray:
        return self.factor * kronecker_product_literal(*self.As)

    def __repr__(self):
        return 'KroneckerProduct({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))

    def __str__(self):
        return 'KroneckerProduct({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))


class KroneckerSum(KroneckerOperator):
    """
    Used to represent the object (A1 ⊕ A2 ⊕ ... ⊕ AN), that is the Kronecker sum of N square matrices.
    """

    def __init__(self, *As):
        check_valid_matrices(*As)
        self.As = As
        self.ndim = len(As)
        self.shapes = tuple(A.shape[0] for A in As)
        N = int(np.prod(self.shapes))
        self.shape = (N, N)

    def __copy__(self):
        new = KroneckerSum(*[A for A in self.As])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = KroneckerSum(*[A.copy() for A in self.As])
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    def operate(self, other: ndarray) -> ndarray:

        # handle when other is a vector
        if other.ndim == 1:
            return self.factor * vec(multiply_tensor_sum(ten(other, shape=tuple(reversed(self.shapes))), *self.As))

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == self.shape[1]:
            out = np.zeros_like(other)
            for i in range(other.shape[1]):
                out[:, i] = vec(multiply_tensor_sum(ten(other[:, i], shape=tuple(reversed(self.shapes))), *self.As))
            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * multiply_tensor_sum(other, *self.As)

    @property
    def T(self):
        return self.factor * KroneckerSum(*[A.T for A in self.As])

    def conj(self):
        return self.factor * KroneckerSum(*[A.conj() for A in self.As])

    def to_array(self) -> ndarray:
        return self.factor * kronecker_sum_literal(*self.As)

    def __repr__(self):
        return 'KroneckerSum({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))

    def __str__(self):
        return 'KroneckerSum({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))


class KroneckerDiag(KroneckerOperator):
    """
    Used to represent a general diagonal matrix of size N1 x N2 x ... x NN
    """

    def __init__(self, A: ndarray):
        """
        Initialise with a tensor of shape (Nn, ..., N1)
        """

        assert isinstance(A, ndarray)
        assert A.ndim > 1, 'The operator diagonal A should be in tensor format, but it is in vector format'

        self.A = A.astype(float)
        self.ndim = A.ndim
        self.shapes = tuple(reversed(A.shape))
        N = int(np.prod(self.shapes))
        self.shape = (N, N)

    def __copy__(self):
        new = KroneckerDiag(self.A)
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = KroneckerDiag(self.A.copy())
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        new = KroneckerDiag(self.A ** power)
        new.factor = self.factor ** power
        return new

    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        # in this case, if other is another KroneckerDiag, we can get a simpler representation
        if isinstance(other, KroneckerDiag):
            check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerDiag(self.A * other.A)

        else:
            return super().__matmul__(other)

    def operate(self, other: ndarray) -> ndarray:

        # handle when other is a vector
        if other.ndim == 1:
            return self.factor * vec(self.A) * other

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == self.shape[1]:
            out = np.zeros_like(other)
            for i in range(other.shape[1]):
                out[:, i] = vec(self.A) * other[:, i]
            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * self.A * other

    def inv(self):
        return self.factor * KroneckerDiag(1 / self.A)

    @property
    def T(self):
        return self

    def conj(self):
        return self.factor * KroneckerDiag(self.A.conj())

    def to_array(self) -> ndarray:
        return self.factor * np.diag(vec(self.A))

    def __repr__(self):
        return 'KroneckerDiag({})'.format(' ⊗ '.join([str(i) for i in reversed(self.A.shape)]))

    def __str__(self):
        return 'KroneckerDiag({})'.format(' ⊗ '.join([str(i) for i in reversed(self.A.shape)]))


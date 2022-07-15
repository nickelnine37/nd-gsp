from numpy import ndarray
from typing import Union
from numpy.linalg import inv

from utils.linalg import vec, tensor_product, tensor_product_of_sum, mat
from scipy.sparse import spmatrix

"""
The classes in this file represenent different types of Kronecker-based operators. Their main purpose is to 
implement __matmul__ in an efficient way, so that we can use the @ syntax without incurring the computational 
cost of performing literal matrix multiplication. 

------ Example -----
A = SomeKroneckerClass(A1, A2, A3)
X = randn(N3, N2, N1)
print(A @ X)
---------------------

These objects are designed to represent valid tensor operators, therefore we restrict ourselves to square 
matrices. 

"""


class KroneckerBase:
    """
    Abstract base class defining the behaviour of Kronecker-type objects
    """

    def __matmul__(self, other: Union['KroneckerBase', ndarray]) -> Union['KroneckerBase', ndarray]:
        """
        All inheriting classes should implement this method
        """
        raise NotImplementedError

    def __rmatmul__(self, other: Union['KroneckerBase', ndarray]) -> Union['KroneckerBase', ndarray]:
        return (self.T @ other.T).T

    def quadratic_form(self, X: ndarray):
        """
        Compute the quadratic form vec(X).T @ self @ vec(X)
        """
        assert isinstance(X, ndarray)
        return (X * (self @ X)).sum()

    def inv(self):
        """
        Inverse method. Use with caution.
        """
        raise NotImplementedError

    @property
    def T(self):
        """
        Transpose property
        """
        raise NotImplementedError


def check_valid_operators(*As):
    assert all(isinstance(A, (ndarray, spmatrix)) for A in As)
    assert all(A.ndim == 2 for A in As)
    assert all(A.shape[0] == A.shape[1] for A in As)


class KroneckerProduct(KroneckerBase):
    """
    Used to represent the object (A1 ⊗ A2 ⊗ ... ⊗ AN), that is the Kronecker product of N square matrices.
    """

    def __init__(self, *As):
        check_valid_operators(*As)
        self.As = As
        self.ndim = len(As)
        self.shapes = [A.shape[0] for A in As]

    def __matmul__(self, other: Union[KroneckerBase, ndarray]) -> Union[KroneckerBase, ndarray]:

        if isinstance(other, KroneckerProduct):
            assert len(self.As) == len(other.As)
            assert all([A1.shape == A2.shape for A1, A2 in zip(self.As, other.As)])
            return KroneckerProduct(*[A1 @ A2 for A1, A2 in zip(self.As, other.As)])

        if isinstance(other, ndarray):
            if other.ndim == 1:
                return vec(tensor_product(mat(other, shape=tuple(reversed(self.shapes))), *self.As))
            else:
                assert other.ndim == self.ndim
                return tensor_product(other, *self.As)

        if isinstance(other, KroneckerBase):
            return KroneckerCoposite(self, other)

        else:
            raise ValueError('other should be a ndarray or a Kronecker type')

    def inv(self):
        return KroneckerProduct(*[inv(A) for A in self.As])

    @property
    def T(self):
        return KroneckerProduct(*[A.T for A in self.As])

    def __repr__(self):
        return 'KroneckerProduct({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))

    def __str__(self):
        return 'KroneckerProduct({})'.format(' ⊗ '.join([str(len(A)) for A in self.As]))


class KroneckerSum(KroneckerBase):
    """
    Used to represent the object (A1 ⊕ A2 ⊕ ... ⊕ AN), that is the Kronecker sum of N square matrices.
    """

    def __init__(self, *As):
        check_valid_operators(*As)
        self.As = As
        self.ndim = len(As)
        self.shapes = [A.shape[0] for A in As]

    def __matmul__(self, other: Union[KroneckerBase, ndarray]) -> Union[KroneckerBase, ndarray]:

        if isinstance(other, ndarray):
            if other.ndim == 1:
                return vec(tensor_product_of_sum(mat(other, shape=tuple(reversed(self.shapes))), *self.As))
            else:
                assert other.ndim == self.ndim
                return tensor_product_of_sum(other, *self.As)

        if isinstance(other, KroneckerBase):
            return KroneckerCoposite(self, other)

        else:
            raise ValueError('other should be a ndarray or a Kronecker type')

    @property
    def T(self):
        return KroneckerSum(*[A.T for A in self.As])

    def __repr__(self):
        return 'KroneckerSum({})'.format(' ⊕ '.join([str(len(A)) for A in self.As]))

    def __str__(self):
        return 'KroneckerSum({})'.format(' ⊕ '.join([str(len(A)) for A in self.As]))


class KroneckerDiag(KroneckerBase):
    """
    Used to represent a general diagonal matrix of size N1 x N2 x ... x NN
    """

    def __init__(self, A: ndarray):
        """
        Initialise with a tensor of shape (NN, ..., N1)
        """

        assert isinstance(A, ndarray)
        assert A.ndim > 1, 'The operator diagonal A should be in tensor format, but it is in vector format'

        self.A = A
        self.ndim = A.ndim

    def __matmul__(self, other: Union[KroneckerBase, ndarray]) -> Union[KroneckerBase, ndarray]:

        if isinstance(other, ndarray):
            if other.ndim == 1:
                return vec(self.A) * other
            else:
                assert other.ndim == self.ndim
                return self.A * other

        elif isinstance(other, KroneckerBase):
            return KroneckerCoposite(self, other)

        else:
            raise ValueError('other should be a ndarray or a Kronecker type')

    def inv(self):
        return KroneckerDiag(1 / self.A)

    @property
    def T(self):
        return self

    def __repr__(self):
        return 'KroneckerDiag{}'.format(self.A.shape)

    def __str__(self):
        return 'KroneckerDiag{}'.format(self.A.shape)


class KroneckerCoposite(KroneckerBase):
    """
    Used to represent a chain of Kronecker objects
    """

    def __init__(self, *krons):
        assert all(isinstance(A, KroneckerBase) for A in krons), 'All matrices passed should be of Kronecker type'
        self.krons = sum([[A] if not isinstance(A, KroneckerCoposite) else [AA for AA in A.krons] for A in krons], [])


    def __matmul__(self, other: Union[KroneckerBase, ndarray]) -> Union[KroneckerBase, ndarray]:

        if isinstance(other, ndarray):
            out = self.krons[-1] @ other
            for kron in reversed(self.krons[:-1]):
                out = kron @ out
            return out

        elif isinstance(other, KroneckerBase):
            return KroneckerCoposite(*self.krons, other)

    def inv(self):
        return KroneckerCoposite(*[kron.inv() for kron in self.krons[::-1]])

    @property
    def T(self):
        return KroneckerCoposite(*[kron.T for kron in self.krons[::-1]])

    def __repr__(self):
        return 'KroneckerCoposite({})'.format(', '.join([str(kron) for kron in self.krons]))

    def __str__(self):
        return 'KroneckerCoposite({})'.format(', '.join([str(kron) for kron in self.krons]))

import numpy as np
from numpy import ndarray
from typing import Union
from numpy.linalg import inv

from utils.linalg import vec, multiply_tensor_product, multiply_tensor_sum, ten, kronecker_product_literal, kronecker_sum_literal
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


class KroneckerOperator:
    """
    Abstract base class defining the behaviour of Kronecker-type objects
    """

    __array_priority__ = 10     # increase priority of class, so it takes precedence when mixing matrix multiplications with ndarrays

    def __matmul__(self, other: Union['KroneckerOperator', ndarray]) -> Union['KroneckerOperator', ndarray]:
        """
        All inheriting classes should implement this method
        """
        raise NotImplementedError

    def __rmatmul__(self, other: Union['KroneckerOperator', ndarray]) -> Union['KroneckerOperator', ndarray]:
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


class KroneckerProduct(KroneckerOperator):
    """
    Used to represent the object (A1 ⊗ A2 ⊗ ... ⊗ AN), that is the Kronecker product of N square matrices.
    """

    def __init__(self, *As):
        check_valid_operators(*As)
        self.As = As
        self.ndim = len(As)
        self.shapes = [A.shape[0] for A in As]

    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        if isinstance(other, KroneckerProduct):
            assert len(self.As) == len(other.As)
            assert all([A1.shape == A2.shape for A1, A2 in zip(self.As, other.As)])
            return KroneckerProduct(*[A1 @ A2 for A1, A2 in zip(self.As, other.As)])

        if isinstance(other, ndarray):
            if other.ndim == 1:
                return vec(multiply_tensor_product(ten(other, shape=tuple(reversed(self.shapes))), *self.As))
            else:
                assert other.ndim == self.ndim
                return multiply_tensor_product(other, *self.As)

        if isinstance(other, KroneckerOperator):
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


class KroneckerSum(KroneckerOperator):
    """
    Used to represent the object (A1 ⊕ A2 ⊕ ... ⊕ AN), that is the Kronecker sum of N square matrices.
    """

    def __init__(self, *As):
        check_valid_operators(*As)
        self.As = As
        self.ndim = len(As)
        self.shapes = [A.shape[0] for A in As]

    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        if isinstance(other, ndarray):
            if other.ndim == 1:
                return vec(multiply_tensor_sum(ten(other, shape=tuple(reversed(self.shapes))), *self.As))
            else:
                assert other.ndim == self.ndim
                return multiply_tensor_sum(other, *self.As)

        if isinstance(other, KroneckerOperator):
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


class KroneckerDiag(KroneckerOperator):
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

    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        if isinstance(other, ndarray):
            if other.ndim == 1:
                return vec(self.A) * other
            else:
                assert other.ndim == self.ndim
                return self.A * other

        elif isinstance(other, KroneckerOperator):
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


class KroneckerCoposite(KroneckerOperator):
    """
    Used to represent a chain of Kronecker objects mattrix multiplied together
    """

    def __init__(self, *krons):
        assert all(isinstance(A, KroneckerOperator) for A in krons), 'All matrices passed should be of Kronecker type'
        self.krons = sum([[A] if not isinstance(A, KroneckerCoposite) else [AA for AA in A.krons] for A in krons], [])


    def __matmul__(self, other: Union[KroneckerOperator, ndarray]) -> Union[KroneckerOperator, ndarray]:

        if isinstance(other, ndarray):
            out = self.krons[-1] @ other
            for kron in reversed(self.krons[:-1]):
                out = kron @ out
            return out

        elif isinstance(other, KroneckerOperator):
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



def _run_tests(seed: int=0):

    np.random.seed(seed)
    N1 = 3; N2 = 4; N3 = 5
    A1 = np.random.randn(N1, N1)
    A2 = np.random.randn(N2, N2)
    A3 = np.random.randn(N3, N3)
    X = np.random.randn(N3, N2, N1)
    d = np.random.randn(N1 * N2 * N3)

    def test_kronecker_product():

        kp_literal = kronecker_product_literal(A1, A2, A3)
        kp_optimised = KroneckerProduct(A1, A2, A3)

        assert np.allclose(kp_literal @ vec(X), kp_optimised @ vec(X))                      # test forward matrix multiplication
        assert np.allclose(vec(X) @ kp_literal, vec(X) @ kp_optimised)                      # test backward matrix multiplication
        assert np.isclose(vec(X) @ kp_literal @ vec(X), kp_optimised.quadratic_form(X))     # test quadratic form

    def test_kronecker_sum():

        ks_literal = kronecker_sum_literal(A1, A2, A3)
        ks_optimised = KroneckerSum(A1, A2, A3)

        assert np.allclose(ks_literal @ vec(X), ks_optimised @ vec(X))                      # test forward matrix multiplication
        assert np.allclose(vec(X) @ ks_literal, vec(X) @ ks_optimised)                      # test backward matrix multiplication
        assert np.isclose(vec(X) @ ks_literal @ vec(X), ks_optimised.quadratic_form(X))     # test quadratic form

    def test_kronecker_diag():

        kd_literal = np.diag(d)
        kd_optimised = KroneckerDiag(ten(d, like=X))

        assert np.allclose(kd_literal @ vec(X), kd_optimised @ vec(X))                      # test forward matrix multiplication
        assert np.allclose(vec(X) @ kd_literal, vec(X) @ kd_optimised)                      # test backward matrix multiplication
        assert np.isclose(vec(X) @ kd_literal @ vec(X), kd_optimised.quadratic_form(X))     # test quadratic form

    def test_kronecker_composite():

        kc_literal = kronecker_sum_literal(A1, A2, A3) @  np.diag(d) @ kronecker_product_literal(A1, A2, A3)
        kc_optimised = KroneckerCoposite(KroneckerSum(A1, A2, A3), KroneckerDiag(ten(d, like=X)), KroneckerProduct(A1, A2, A3))

        assert np.allclose(kc_literal @ vec(X), kc_optimised @ vec(X))                      # test forward matrix multiplication
        assert np.allclose(vec(X) @ kc_literal, vec(X) @ kc_optimised)                      # test backward matrix multiplication
        assert np.isclose(vec(X) @ kc_literal @ vec(X), kc_optimised.quadratic_form(X))     # test quadratic form

    test_kronecker_product()
    test_kronecker_sum()
    test_kronecker_diag()
    test_kronecker_composite()

    print('All tests passed')


if __name__ == '__main__':

    _run_tests(seed=0)


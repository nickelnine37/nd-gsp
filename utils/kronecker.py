import numpy as np
from numpy import ndarray
from typing import Union
from numpy.linalg import inv

from utils.linalg import vec, multiply_tensor_product, multiply_tensor_sum, ten, kronecker_product_literal, kronecker_sum_literal
from scipy.sparse import spmatrix
from numbers import Number


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
    Base class defining the behaviour of Kronecker-type operators
    """

    __array_priority__ = 10     # increase priority of class, so it takes precedence when mixing matrix multiplications with ndarrays
    shape: tuple = None         # full (N, N) operator shape
    shapes: tuple = None        # (N1, N2, ...) individual shapes
    ndim: int = None            # number of dimensions
    factor: Number = 1          # a scalar factor multiplying the whole operator 

    def __add__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':

        if not isinstance(other, KroneckerOperator):
            raise TypeError('Konecker operators can only be added to other Kronecker operators')

        return SumChain(self, other)

    def __radd__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':
        return self.__add__(other)

    def __sub__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':
        return self.__add__((-1) * other)
    
    def __mul__(self, other):
        if isinstance(other, Number):
            self.factor *= other
        elif isinstance(other, KroneckerOperator):
            raise TypeError('Only KroneckerProducts can be multipled together element-wise')
        else:
            raise TypeError('Kronecker operators can only be scaled by a number')
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    def __matmul__(self, other: Union['KroneckerOperator', ndarray]) -> Union['KroneckerOperator', ndarray]:

        if isinstance(other, ndarray):
            return self.operate(other)

        elif isinstance(other, KroneckerOperator):
            return ProductChain(self, other)

        else:
            raise TypeError('Both objects in the matrix product must be Kronecker Operators')

    def __rmatmul__(self, other: Union['KroneckerOperator', ndarray]) -> Union['KroneckerOperator', ndarray]:
        return (self.T @ other.T).T

    def __pos__(self):
        return self

    def __neg__(self):
        return (-1) * self

    def operate(self, other: ndarray) -> ndarray:
        raise NotImplementedError

    def quadratic_form(self, X: ndarray) -> float:
        """
        Compute the quadratic form vec(X).T @ self @ vec(X)
        """

        if not isinstance(X, ndarray):
            raise TypeError

        return (X * (self @ X)).sum()

    def inv(self):
        """
        Inverse method. Use with caution.
        """
        return NotImplemented

    @property
    def T(self):
        """
        Transpose property
        """
        return NotImplemented


def check_valid_matrices(*As):
    assert all(isinstance(A, (ndarray, spmatrix)) for A in As)
    assert all(A.ndim == 2 for A in As)
    assert all(A.shape[0] == A.shape[1] for A in As)


def check_operators_consistent(*operators):
    assert all(isinstance(A, KroneckerOperator) for A in operators), f'All operators in this chain must be consistent, but they have types {[type(operator) for operator in operators]} respectively'
    assert all(op1.shape == op2.shape for op1, op2 in zip(operators[1:], operators[:-1])), f'All operators in this chain should have the same shape, but they have shapes {[operator.shape for operator in operators]} respectively'
    

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

        if other.ndim == 1:
            return self.factor * vec(multiply_tensor_product(ten(other, shape=tuple(reversed(self.shapes))), *self.As))
        else:
            return self.factor * multiply_tensor_product(other, *self.As)

    def inv(self):
        return self.factor * KroneckerProduct(*[inv(A) for A in self.As])

    @property
    def T(self):
        return self.factor * KroneckerProduct(*[A.T for A in self.As])

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

    def operate(self, other: ndarray) -> ndarray:
        
        if other.ndim == 1:
            return self.factor * vec(multiply_tensor_sum(ten(other, shape=tuple(reversed(self.shapes))), *self.As))
        else:
            return self.factor * multiply_tensor_sum(other, *self.As)
        
    @property
    def T(self):
        return self.factor * KroneckerSum(*[A.T for A in self.As])

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
        Initialise with a tensor of shape (Nn, ..., N1)
        """

        assert isinstance(A, ndarray)
        assert A.ndim > 1, 'The operator diagonal A should be in tensor format, but it is in vector format'

        self.A = A
        self.ndim = A.ndim
        self.shapes = tuple(reversed(A.shape))
        N = int(np.prod(self.shapes))
        self.shape = (N, N)

    def operate(self, other: ndarray) -> ndarray:
        
        if other.ndim == 1:
            return self.factor * vec(self.A) * other
        else:
            return self.factor * self.A * other

    def inv(self):
        return self.factor * KroneckerDiag(1 / self.A)

    @property
    def T(self):
        return self

    def __repr__(self):
        return 'KroneckerDiag{}'.format(self.A.shape)

    def __str__(self):
        return 'KroneckerDiag{}'.format(self.A.shape)


class SumChain(KroneckerOperator):
    """
    Used to represent a chain of Kronecker objects summed together
    """

    def __init__(self, *operators):
        check_operators_consistent(*operators)
        self.chain = operators
        self.shape = self.chain[0].shape
        self.shapes = self.chain[0].shapes

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * sum(operator.operate(other) for operator in self.chain)

    def inv(self):
        raise NotImplementedError

    @property
    def T(self):
        return self.factor * SumChain(*[operator.T for operator in self.chain])

    def __repr__(self):
        return 'SumChain({})'.format(', '.join([str(operator) for operator in self.chain]))

    def __str__(self):
        return 'SumChain({})'.format(', '.join([str(operator) for operator in self.chain]))


class ProductChain(KroneckerOperator):
    """
    Used to represent a chain of Kronecker objects matrix-multiplied together
    """

    def __init__(self, *operators):
        check_operators_consistent(*operators)
        self.chain = operators
        self.shape = self.chain[0].shape
        self.shapes = self.chain[0].shapes
        
    def operate(self, other: ndarray) -> ndarray:

        out = self.chain[-1] @ other
        for operator in reversed(self.chain[:-1]):
            out = operator @ out
        return self.factor * out
        
    def inv(self):
        return self.factor * ProductChain(*[operator.inv() for operator in reversed(self.chain)])

    @property
    def T(self):
        return self.factor * ProductChain(*[operator.T for operator in reversed(self.chain)])

    def __repr__(self):
        return 'ProductChain({})'.format(', '.join([str(operator) for operator in self.chain]))

    def __str__(self):
        return 'ProductChain({})'.format(', '.join([str(operator) for operator in self.chain]))



def _run_tests(seed: int=1):

    np.random.seed(seed)
    N1 = 3; N2 = 4; N3 = 5
    A1 = np.random.randn(N1, N1)
    A2 = np.random.randn(N2, N2)
    A3 = np.random.randn(N3, N3)
    X = np.random.randn(N3, N2, N1)
    D = np.random.randn(N3, N2, N1)

    def run_assertions(literal, optimised):

        assert np.allclose(literal @ vec(X), optimised @ vec(X))                      # test forward matrix multiplication
        assert np.allclose(vec(X) @ literal, vec(X) @ optimised)                      # test backward matrix multiplication
        assert np.isclose(vec(X) @ literal @ vec(X), optimised.quadratic_form(X))     # test quadratic form

    def test_kronecker_product():

        literal = kronecker_product_literal(A1, A2, A3)
        optimised = KroneckerProduct(A1, A2, A3)
        run_assertions(literal, optimised)

    def test_kronecker_sum():

        literal = kronecker_sum_literal(A1, A2, A3)
        optimised = KroneckerSum(A1, A2, A3)
        run_assertions(literal, optimised)

    def test_kronecker_diag():

        literal = np.diag(vec(D))
        optimised = KroneckerDiag(D)
        run_assertions(literal, optimised)

    def test_product_chain():

        literal = kronecker_sum_literal(A1, A2, A3) @  np.diag(vec(D)) @ kronecker_product_literal(A1, A2, A3)
        optimised = ProductChain(KroneckerSum(A1, A2, A3), KroneckerDiag(D), KroneckerProduct(A1, A2, A3))
        run_assertions(literal, optimised)

    def test_sum_chain():

        literal = kronecker_sum_literal(A1, A2, A3) + np.diag(vec(D)) + kronecker_product_literal(A1, A2, A3)
        optimised = SumChain(KroneckerSum(A1, A2, A3), KroneckerDiag(D), KroneckerProduct(A1, A2, A3))
        run_assertions(literal, optimised)

    def test_operator_multiplcation():

        literal = kronecker_sum_literal(A1, A2, A3) @ np.diag(vec(D)) @ kronecker_product_literal(A1, A2, A3)
        optimised = KroneckerSum(A1, A2, A3) @ KroneckerDiag(D) @ KroneckerProduct(A1, A2, A3)
        run_assertions(literal, optimised)

    def test_operator_addition():

        literal = kronecker_sum_literal(A1, A2, A3) + np.diag(vec(D)) + kronecker_product_literal(A1, A2, A3)
        optimised = KroneckerSum(A1, A2, A3) + KroneckerDiag(D) + KroneckerProduct(A1, A2, A3)
        run_assertions(literal, optimised)

    def test_operator_subtraction():

        literal = np.diag(vec(D)) - kronecker_product_literal(A1, A2, A3)
        optimised = KroneckerDiag(D) - KroneckerProduct(A1, A2, A3)
        run_assertions(literal, optimised)

    def test_operator_scaling():

        literal = 2 * kronecker_sum_literal(A1, A2, A3) + np.diag(vec(D)) / 5 - (13 / 11) * kronecker_product_literal(A1, A2, A3)
        optimised = 2 * KroneckerSum(A1, A2, A3) + KroneckerDiag(D) / 5 - (13 / 11) * KroneckerProduct(A1, A2, A3)
        run_assertions(literal, optimised)

    def test_kronecker_hadamard():

        literal = 2 * kronecker_product_literal(A1, A2, A3) * 4 * kronecker_product_literal(A1, A2, A3).T
        optimised = 2 * KroneckerProduct(A1, A2, A3) * 4 * KroneckerProduct(A1, A2, A3).T
        run_assertions(literal, optimised)

    test_kronecker_product()
    test_kronecker_sum()
    test_kronecker_diag()
    test_product_chain()
    test_sum_chain()
    test_operator_multiplcation()
    test_operator_addition()
    test_operator_subtraction()
    test_operator_scaling()
    test_kronecker_hadamard()

    print('All tests passed')


if __name__ == '__main__':

    _run_tests(seed=0)


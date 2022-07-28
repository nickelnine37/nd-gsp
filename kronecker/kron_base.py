import numpy as np
from numpy import ndarray
from typing import Union

from scipy.sparse import spmatrix
from numbers import Number
from utils.linalg import kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal, vec

"""
The class in this file is a base class for all Kronecker operators. Kronecker operators represent large matrices in a
compact form, and perform all multiplications onto vectors lazily and efficiently. Composite operators can be created
by treating Kronecker operators as if they are NumPy matrices. All operators support:

    * addition
    * matrix multiplication
    * multiplication/division by a scalar
    * summing along axes 0, 1 or both
    * transposing/conjugate

using +, @, * .sum() and .T respectively. Some further behaviours for certian operator types are implemented in the subclasses. 
    
"""


class KroneckerOperator:
    """
    Base class defining the behaviour of Kronecker-type operators. It should not be instantiated directly.
    """

    __array_priority__ = 10     # increase priority of class, so it takes precedence when mixing matrix multiplications with ndarrays
    factor: float = 1.0         # a scalar factor multiplying the whole operator
    shape: tuple = None         # full (N, N) operator shape
    state = {}                  # contains any other state that the

    def __copy__(self) -> 'KroneckerOperator':
        """
        Create a shallow copy of a Kronecker object. This does not copy any of the underlying arrays, but means,
        for example, we can have kronecker objects with the same underlying arrays but different factors. This
        needs to be implemented by subclasses.
        """
        raise NotImplementedError

    def __deepcopy__(self, memodict={}) -> 'KroneckerOperator':
        """
        Create a deep copy of a Kronecker object. This copies the data in the underlying arrays to create
        a totally independent object. This needs to be implemented by subclasses.
        """
        raise NotImplementedError

    def __add__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':
        """
        Overload the addition method. This is used to sum together KroneckerOperators and as such
        `other` must be an instance of a KroneckerOperator, and not an array or other numeric type.
        """

        from kronecker.kron_composite import OperatorSum

        if not isinstance(other, KroneckerOperator):
            raise TypeError('Konecker operators can only be added to other Kronecker operators')

        return OperatorSum(self, other)

    def __radd__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':
        """
        Order does not matter, so return __add__
        """
        return self.__add__(other)

    def __sub__(self, other: 'KroneckerOperator') -> 'KroneckerOperator':
        """
        Overload the subtraction method. Simply scale `other` by negative one and add
        """
        return self.__add__((-1.0) * other)
    
    def __mul__(self, other: Union['KroneckerOperator', Number]) -> 'KroneckerOperator':
        """
        Multiply the linear operator element-wise. As with numpy arrays, the * operation defaults to element-wise
        (Hadamard) multiplication, not matrix multiplication. For numbers, this is a simple scaler multiple. For
        Kronecker objects, we can only define the behaviour efficiently for KroneckerProducts and KroneckerDiags,
        which is implemented in the respective subclass.
        """

        if isinstance(other, Number):
            # create a copy of the object rather than mutating the factor directly, which is cleaner and leads to less unexpected behaviour
            new = self.copy()
            new.factor = self.factor * other
            return new

        elif isinstance(other, KroneckerOperator):
            raise TypeError('Only KroneckerProducts and KroneckerDiags can be multipled together element-wise')

        else:
            raise TypeError('General Kronecker operators can only be scaled by a number')

    def __rmul__(self, other: Union['KroneckerOperator', Number]) -> 'KroneckerOperator':
        """
        Hadamard and scaler multiples are commutitive
        """
        return self.__mul__(other)

    def __truediv__(self, other: Number) -> 'KroneckerOperator':
        """
        Self-explanatory, but only works for numbers.
        """
        return self.__mul__(1.0 / other)

    def __pow__(self, power, modulo=None):
        """
        Element-wise power operation. Only works for KroneckerProducts and KroneckerDiags.
        """
        raise NotImplementedError('Element-wise power operation only works for KroneckerProducts and KroneckerDiags')

    def __matmul__(self, other: Union['KroneckerOperator', ndarray]) -> Union['KroneckerOperator', ndarray]:
        """
        Overload the matrix multiplication method to use these objects with the @ operator.
        """

        from kronecker.kron_composite import OperatorProduct

        if isinstance(other, ndarray):
            return self.operate(other)

        elif isinstance(other, KroneckerOperator):
            return OperatorProduct(self, other)

        else:
            raise TypeError('Both objects in the matrix product must be Kronecker Operators')

    def __rmatmul__(self, other: Union['KroneckerOperator', ndarray]) -> Union['KroneckerOperator', ndarray]:
        """
        Define reverve matrix multiplication in terms of transposes
        """
        return (self.T @ other.T).T

    def __pos__(self):
        """
        +Obj == Obj
        """
        return self

    def __neg__(self):
        """
        -Obj == (-1) * Obj
        """
        return (-1) * self

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        return f'KroneckerOperator{self.shape}'

    def __repr__(self):
        return f'KroneckerOperator{self.shape}'

    def __array_ufunc__(self, method, *inputs, **kwargs):
        """
        Override the numpy implementation of matmul, so that we can also use
        this funcion rather than just the @ operator.

        E.g.
            KroneckerProduct(A, B, C) @ vec(X) == np.matmul(KroneckerProduct(A, B, C), vec(X))

        Note that
            KroneckerProduct(A, B, C) @ X !=  np.matmul(KroneckerProduct(A, B, C), X)

        and that it does not work with np.dot()
        """


        if method is np.matmul:

            A, B = inputs[1], inputs[2]

            if A is self:
                return self.__matmul__(B)

            if B is self:
                return self.__rmatmul__(A)

        else:
            raise NotImplementedError

    def operate(self, other: ndarray) -> ndarray:
        """
        This key method should describe how the Kronecker object acts on a Tensor/vector. This is where subclasses should
        implement their efficient versions of matrix-vector multiplication.
        """
        raise NotImplementedError

    def quadratic_form(self, X: ndarray) -> float:
        """
        Compute the quadratic form vec(X).T @ self @ vec(X)
        """

        if not isinstance(X, ndarray):
            raise TypeError

        return (X * (self @ X)).sum()

    def sum(self, axis=None):
        """
        Sum the operator along one axis as if it is a matrix. Or None for total sum.
        """

        if axis is not None and (axis > 1 or axis < -1):
            raise ValueError('Axis should be -1, 0, 1 or None')

        else:
            ones = np.ones(self.shape[0])

            if axis is None:
                return self.quadratic_form(ones)

            elif axis == 1 or axis == -1:
                return self @ ones

            elif axis == 0:
                return self.T @ ones

            else:
                raise ValueError('Axis should be -1, 0, 1 or None')

    def inv(self):
        """
        Inverse method. Use with caution.
        """
        raise NotImplementedError

    @property
    def T(self):
        """
        Return a copy of the object transposed.
        """
        raise NotImplementedError

    def conj(self):
        """
        Return the complex conjugate of the operator
        """
        raise NotImplementedError

    def to_array(self) -> ndarray:
        """
        Turn into a literal array. Use with caution!
        """

        raise NotImplementedError

    def copy(self):
        return self.__copy__()

    def deepcopy(self):
        return self.__deepcopy__()


def check_valid_matrices(*As) -> bool:

    assert all(isinstance(A, (ndarray, spmatrix)) for A in As)
    assert all(A.ndim == 2 for A in As)
    assert all(A.shape[0] == A.shape[1] for A in As)

    return True

def check_operators_consistent(A: KroneckerOperator, B: KroneckerOperator) -> bool:

    assert all(isinstance(C, KroneckerOperator) for C in [A, B]), f'All operators in this chain must be consistent, but they have types {type(A)} and {type(B)} respectively'
    assert A.shape == B.shape, f'All operators in this chain should have the same shape, but they have shapes {A.shape} and {B.shape} respectively'

    return True


def check_blocks_consistent(blocks: list):
    """
    Check the blocks, which are provided as input to KroneckerBlock and KroneckerBlockDiag are consistent
    """

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


def generate_test_data(seed: int=0):
    """
    Generate random data for testing purposes
    """

    from kronecker.kron_operators import KroneckerProduct, KroneckerDiag, KroneckerSum

    np.random.seed(seed)

    N1 = 6
    N2 = 5
    N3 = 4
    N4 = 3
    K = 5

    A1 = np.random.randn(N1, N1)
    A2 = np.random.randn(N2, N2)
    A3 = np.random.randn(N3, N3)
    A4 = np.random.randn(N4, N4)
    D = np.random.randn(N4, N3, N2, N1)

    X = np.random.randn(N4, N3, N2, N1)
    Y = np.random.randn(N4, N3, N2, N1)
    Q = np.random.randn(N4 * N3 * N2 * N1, K)

    # create actual array structures
    kp_literal = kronecker_product_literal(A1, A2, A3, A4)
    ks_literal = kronecker_sum_literal(A1, A2, A3, A4)
    kd_literal = kronecker_diag_literal(D)

    kp_optimised = KroneckerProduct(A1, A2, A3, A4)
    ks_optimised = KroneckerSum(A1, A2, A3, A4)
    kd_optimised = KroneckerDiag(D)

    # print('kp_optimised', kp_optimised.to_array())
    # print('ks_optimised', ks_optimised.to_array())

    return X, Y, Q, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised


def run_assertions(X: ndarray, P: ndarray, literal: ndarray, optimised: KroneckerOperator):
    """
    Assert that the ndarray `literal` and the KroneckerOperator`optimised` behave in the exact same
    way when applied to the tensor X, and matrix of vectors P.
    """

    # print('literal2', literal)
    # print('optimised2', optimised.to_array())

    # test literal conversion
    assert np.allclose(literal, optimised.to_array())

    # test with @ operator
    assert np.allclose(literal @ vec(X), optimised @ vec(X))                      # test forward matrix multiplication
    assert np.allclose(vec(X) @ literal, vec(X) @ optimised)                      # test backward matrix multiplication
    assert np.isclose(vec(X) @ literal @ vec(X), vec(X) @ optimised @ vec(X))     # test quadratic form

    # # test with np.matmul
    # assert np.allclose(np.matmul(literal, vec(X)), np.matmul(optimised, vec(X)))
    # assert np.allclose(np.matmul(vec(X), literal), np.matmul(vec(X), optimised))
    # assert np.isclose(np.matmul(vec(X), np.matmul(literal, vec(X))), np.matmul(vec(X), np.matmul(optimised, vec(X))))

    # test multiplication onto a data matrix
    assert np.allclose(literal @ P, optimised @ P)
    assert np.allclose(P.T @ literal, P.T @ optimised)
    assert np.allclose(P.T @ literal @ P, P.T @ optimised @ P)

    # test summing operation
    assert np.allclose(literal.sum(0), optimised.sum(0))
    assert np.allclose(literal.sum(1), optimised.sum(1))
    assert np.isclose(literal.sum(), optimised.sum())




#
# def _run_tests(seed: int=1):
#
#     np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)
#
#     np.random.seed(seed)
#
#     N1 = 6
#     N2 = 5
#     N3 = 4
#     N4 = 3
#     K = 5
#
#     A1 = np.random.randn(N1, N1)
#     A2 = np.random.randn(N2, N2)
#     A3 = np.random.randn(N3, N3)
#     A4 = np.random.randn(N4, N4)
#
#     Y = np.random.randn(N4, N3, N2, N1)
#     D = np.random.randn(N4, N3, N2, N1)
#     Q = np.random.randn(N4 * N3 * N2 * N1, K)
#
#     # create actual array structures
#     kp_literal = kronecker_product_literal(A1, A2, A3, A4)
#     ks_literal = kronecker_sum_literal(A1, A2, A3, A4)
#     kd_literal = kronecker_diag_literal(D)
#     kb_literal = np.block([[kp_literal, kd_literal], [np.zeros(kp_literal.shape), ks_literal]])
#     kbd_literal = np.block([[kp_literal, np.zeros(kp_literal.shape)], [np.zeros(kp_literal.shape), ks_literal]])
#
#     # create lazy computation equivelants
#     kp_optimised = KroneckerProduct(A1, A2, A3, A4)
#     ks_optimised = KroneckerSum(A1, A2, A3, A4)
#     kd_optimised = KroneckerDiag(D)
#     kb_optimised = KroneckerBlock([[kp_optimised, kd_optimised], [np.zeros(kp_literal.shape), ks_optimised]])
#     kbd_optimised = KroneckerBlockDiag([kp_optimised, ks_optimised])
#
#
#     def run_regular_assertions(literal: ndarray, optimised: KroneckerOperator):
#
#         X = np.random.randn(N4, N3, N2, N1)
#         P = np.random.randn(N4 * N3 * N2 * N1, K)
#
#         # test with @ operator
#         assert np.allclose(literal @ vec(X), optimised @ vec(X))                      # test forward matrix multiplication
#         assert np.allclose(vec(X) @ literal, vec(X) @ optimised)                      # test backward matrix multiplication
#         assert np.isclose(vec(X) @ literal @ vec(X), vec(X) @ optimised @ vec(X))     # test quadratic form
#
#         # test with np.matmul
#         assert np.allclose(np.matmul(literal, vec(X)), np.matmul(optimised, vec(X)))
#         assert np.allclose(np.matmul(vec(X), literal), np.matmul(vec(X), optimised))
#         assert np.isclose(np.matmul(vec(X), np.matmul(literal, vec(X))), np.matmul(vec(X), np.matmul(optimised, vec(X))))
#
#         # test summing operation
#         assert np.allclose(literal.sum(0), optimised.sum(0))
#         assert np.allclose(literal.sum(1), optimised.sum(1))
#         assert np.isclose(literal.sum(), optimised.sum())
#
#         # test literal conversion
#         assert np.allclose(literal, optimised.to_array())
#
#         # test multiplication onto a data matrix
#         assert np.allclose(literal @ P, optimised @ P)
#         assert np.allclose(P.T @ literal, P.T @ optimised)
#         assert np.allclose(P.T @ literal @ P, P.T @ optimised @ P)
#
#     def run_block_assertions(literal: ndarray, optimised: Union[KroneckerBlock, KroneckerBlockDiag]):
#
#         X1 = np.random.randn(N4 * N3 * N2 * N1 * 2)
#         P = np.random.randn(N4 * N3 * N2 * N1 * 2, K)
#
#         # test with @ operator
#         assert np.allclose(literal @ X1, optimised @ X1)
#         assert np.allclose(X1 @ literal, X1 @ optimised)
#         assert np.isclose(X1 @ literal @ X1, X1 @ optimised @ X1)
#
#         assert np.allclose(literal, optimised.to_array())
#
#         # test multiplication onto a data matrix
#         assert np.allclose(literal @ P, optimised @ P)
#         assert np.allclose(P.T @ literal, P.T @ optimised)
#         assert np.allclose(P.T @ literal @ P, P.T @ optimised @ P)
#
#
#     def test_kronecker_product():
#         run_regular_assertions(kp_literal, kp_optimised)
#
#     def test_kronecker_sum():
#         run_regular_assertions(ks_literal, ks_optimised)
#
#     def test_kronecker_diag():
#         run_regular_assertions(kd_literal, kd_optimised)
#
#     def test_kronecker_block():
#
#         run_block_assertions(kb_literal, kb_optimised)
#         run_block_assertions(kb_literal.T, kb_optimised.T)
#         run_block_assertions(kbd_literal, kbd_optimised)
#         run_block_assertions(kbd_literal.T, kbd_optimised.T)
#
#
#     def test_product_chain():
#
#         literal = ks_literal @  kd_literal @ kp_literal
#         optimised = OperatorProduct(ks_optimised, kd_optimised, kp_optimised)
#         run_regular_assertions(literal, optimised)
#
#     def test_sum_chain():
#
#         literal = ks_literal + kd_literal + kp_literal
#         optimised = OperatorSum(ks_optimised, kd_optimised, kp_optimised)
#         run_regular_assertions(literal, optimised)
#
#     def test_operator_multiplcation():
#
#         literal = ks_literal @ kd_literal @ kp_literal
#         optimised = ks_optimised @ kd_optimised @ kp_optimised
#         run_regular_assertions(literal, optimised)
#
#     def test_operator_addition():
#
#         literal = ks_literal + kd_literal + kp_literal
#         optimised = ks_optimised + kd_optimised + kp_optimised
#         run_regular_assertions(literal, optimised)
#
#     def test_operator_subtraction():
#
#         literal = kd_literal - kp_literal
#         optimised = kd_optimised - kp_optimised
#         run_regular_assertions(literal, optimised)
#
#     def test_operator_scaling():
#
#         literal = 2 * ks_literal + kd_literal / 5 - (13 / 11) * kp_literal
#         optimised = 2 * ks_optimised + kd_optimised / 5 - (13 / 11) * kp_optimised
#         run_regular_assertions(literal, optimised)
#
#     def test_kronecker_hadamard():
#
#         literal = 2 * kp_literal * 4 * kp_literal.T
#         optimised = 2 * kp_optimised * 4 * kp_optimised.T
#         run_regular_assertions(literal, optimised)
#
#     def test_kronecker_pow():
#
#         literal = kp_literal ** 2 + kd_literal ** 3
#         optimised = kp_optimised ** 2 + kd_optimised ** 3
#         run_regular_assertions(literal, optimised)
#
#     def test_assorted_expressions():
#
#         literal1 = (2 * kp_literal.T - ks_literal) @ kp_literal / 2.2
#         literal2 =  -5 * kd_literal @ ks_literal.T + kp_literal @ kp_literal.T
#         literal3 = -kp_literal.T @ ks_literal @ kp_literal
#         literal4 = (kp_literal - ks_literal @  kd_literal).T @ ks_literal
#         literal5 = (kp_literal * kp_literal.T) @  kp_literal.T
#
#         optimised1 = (2 * kp_optimised.T - ks_optimised) @ kp_optimised / 2.2
#         optimised2 =  -5 * kd_optimised @ ks_optimised.T + kp_optimised @ kp_optimised.T
#         optimised3 = -kp_optimised.T @ ks_optimised @ kp_optimised
#         optimised4 = (kp_optimised - ks_optimised @   kd_optimised).T @ ks_optimised
#         optimised5 = (kp_optimised * kp_optimised.T) @  kp_optimised.T
#
#         run_regular_assertions(literal1, optimised1)
#         run_regular_assertions(literal2, optimised2)
#         run_regular_assertions(literal3, optimised3)
#         run_regular_assertions(literal4, optimised4)
#         run_regular_assertions(literal5, optimised5)
#
#     test_kronecker_product()
#     test_kronecker_sum()
#     test_kronecker_diag()
#     test_kronecker_block()
#     test_product_chain()
#     test_sum_chain()
#     test_operator_multiplcation()
#     test_operator_addition()
#     test_operator_subtraction()
#     test_operator_scaling()
#     test_kronecker_hadamard()
#     test_kronecker_pow()
#     test_assorted_expressions()
#
#     print('All tests passed')
#
#
# if __name__ == '__main__':
#
#     _run_tests(seed=0)
#

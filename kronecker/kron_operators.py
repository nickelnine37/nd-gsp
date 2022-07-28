import numpy as np
from numpy import ndarray
from typing import Union
from numpy.linalg import inv

# import kronecker as kron
import kronecker.kron_base
from utils.linalg import vec, multiply_tensor_product, multiply_tensor_sum, ten, kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal



class KroneckerProduct(kronecker.kron_base.KroneckerOperator):
    """
    Used to represent the object (A1 ⊗ A2 ⊗ ... ⊗ AN), that is the Kronecker product of N square matrices.
    """

    def __init__(self, *As):
        """
        Initialise by passing in a sequence of square arrays as Numpy arrays or spmatrices
        """

        kronecker.kron_base.check_valid_matrices(*As)
        self.state = {'As': As}
        N = int(np.prod([A.shape[0] for A in As]))
        self.shape = (N, N)

    def __copy__(self):
        new = KroneckerProduct(*[A for A in self.state['As']])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = KroneckerProduct(*[A.copy() for A in self.state['As']])
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        return self.factor ** power * KroneckerProduct(*[A ** power for A in self.state['As']])

    def __matmul__(self, other: Union[kronecker.kron_base.KroneckerOperator, ndarray]) -> Union[kronecker.kron_base.KroneckerOperator, ndarray]:

        # in this case, if other is another Kronecker product, we can get a simpler representation
        if isinstance(other, KroneckerProduct):

            kronecker.kron_base.check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerProduct(*[A1 @ A2 for A1, A2 in zip(self.state['As'], other.state['As'])])

        # otherwise default to creating an OperatorChain
        else:
            return super().__matmul__(other)

    def __mul__(self, other):

        # kronecker products can be hadamarded against other kronecker products only
        if isinstance(other, KroneckerProduct):

            kronecker.kron_base.check_operators_consistent(self, other)
            return self.factor * other.factor * KroneckerProduct(*[A1 * A2 for A1, A2 in zip(self.state['As'], other.state['As'])])

        # otherwise other should be a scalar, handled in the base class
        else:
            return super().__mul__(other)

    def operate(self, other: ndarray) -> ndarray:

        other = np.squeeze(other)

        # handle when other is a vector
        if other.ndim == 1:
            other_ten = ten(other, shape=tuple(A.shape[0] for A in reversed(self.state['As'])))
            return self.factor * vec(multiply_tensor_product(other_ten, *self.state['As']))

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == len(self):

            out = np.zeros_like(other)

            for i in range(other.shape[1]):
                other_ten = ten(other[:, i], shape=tuple(A.shape[0] for A in reversed(self.state['As'])))
                out[:, i] = vec(multiply_tensor_product(other_ten, *self.state['As']))

            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * multiply_tensor_product(other, *self.state['As'])

    def inv(self):
        return self.factor * KroneckerProduct(*[inv(A) for A in self.state['As']])

    @property
    def T(self):
        return self.factor * KroneckerProduct(*[A.T for A in self.state['As']])

    def conj(self):
        return self.factor * KroneckerProduct(*[A.conj() for A in self.state['As']])

    def to_array(self) -> ndarray:
        return self.factor * kronecker_product_literal(*self.state['As'])

    def __repr__(self):
        return 'KroneckerProduct({})'.format(' ⊗ '.join([str(len(A)) for A in self.state['As']]))

    def __str__(self):
        return 'KroneckerProduct({})'.format(' ⊗ '.join([str(len(A)) for A in self.state['As']]))


class KroneckerSum(kronecker.kron_base.KroneckerOperator):
    """
    Used to represent the object (A1 ⊕ A2 ⊕ ... ⊕ AN), that is the Kronecker sum of N square matrices.
    """

    def __init__(self, *As):

        kronecker.kron_base.check_valid_matrices(*As)
        self.state = {'As': As}
        N = int(np.prod([A.shape[0] for A in As]))
        self.shape = (N, N)

    def __copy__(self):
        new = KroneckerSum(*[A for A in self.state['As']])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = KroneckerSum(*[A.copy() for A in self.state['As']])
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    def operate(self, other: ndarray) -> ndarray:

        other = np.squeeze(other)

        # handle when other is a vector
        if other.ndim == 1:
            other_ten = ten(other, shape=tuple(A.shape[0] for A in reversed(self.state['As'])))
            return self.factor * vec(multiply_tensor_sum(other_ten, *self.state['As']))

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == len(self):

            out = np.zeros_like(other)

            for i in range(other.shape[1]):
                other_ten = ten(other[:, i], shape=tuple(A.shape[0] for A in reversed(self.state['As'])))
                out[:, i] = vec(multiply_tensor_sum(other_ten, *self.state['As']))

            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * multiply_tensor_sum(other, *self.state['As'])

    @property
    def T(self):
        return self.factor * KroneckerSum(*[A.T for A in self.state['As']])

    def conj(self):
        return self.factor * KroneckerSum(*[A.conj() for A in self.state['As']])

    def inv(self):
        raise NotImplementedError

    def to_array(self) -> ndarray:
        return self.factor * kronecker_sum_literal(*self.state['As'])

    def __repr__(self):
        return 'KroneckerSum({})'.format(' ⊗ '.join([str(len(A)) for A in self.state['As']]))

    def __str__(self):
        return 'KroneckerSum({})'.format(' ⊗ '.join([str(len(A)) for A in self.state['As']]))


class KroneckerDiag(kronecker.kron_base.KroneckerOperator):
    """
    Used to represent a general diagonal matrix of size N1 x N2 x ... x NN
    """

    def __init__(self, A: ndarray):
        """
        Initialise with a tensor of shape (Nn, ..., N1)
        """

        assert isinstance(A, ndarray)
        assert A.ndim > 1, 'The operator diagonal A should be in tensor format, but it is in vector format'

        self.state = {'A': A}

        N = int(np.prod(A.shape))
        self.shape = (N, N)

    def __copy__(self):
        new = KroneckerDiag(self.state['A'])
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = KroneckerDiag(self.state['A'].copy())
        new.factor = self.factor
        return new

    def __pow__(self, power, modulo=None):
        new = KroneckerDiag(self.state['A'] ** power)
        new.factor = self.factor ** power
        return new

    def __matmul__(self, other: Union[kronecker.kron_base.KroneckerOperator, ndarray]) -> Union[kronecker.kron_base.KroneckerOperator, ndarray]:

        # in this case, if other is another KroneckerDiag, we can get a simpler representation
        if isinstance(other, KroneckerDiag):

            kronecker.kron_base.check_operators_consistent(self, other)

            return self.factor * other.factor * KroneckerDiag(self.state['A'] * other.state['A'])

        else:
            return super().__matmul__(other)

    def operate(self, other: ndarray) -> ndarray:

        # handle when other is a vector
        if other.ndim == 1:
            return self.factor * vec(self.state['A']) * other

        # handle when other is a matrix of column vectors
        elif other.ndim == 2 and other.shape[0] == len(self):

            out = np.zeros_like(other)

            for i in range(other.shape[1]):
                out[:, i] = vec(self.state['A']) * other[:, i]

            return self.factor * out

        # handle when other is a tensor
        else:
            return self.factor * self.state['A'] * other

    def inv(self):
        return self.factor * KroneckerDiag(1 / self.state['A'])

    @property
    def T(self):
        return self

    def conj(self):
        return self.factor * KroneckerDiag(self.state['A'].conj())

    def to_array(self) -> ndarray:
        return self.factor * kronecker_diag_literal(self.state['A'])

    def __repr__(self):
        return 'KroneckerDiag({})'.format(' ⊗ '.join([str(i) for i in reversed(self.state['A'].shape)]))

    def __str__(self):
        return 'KroneckerDiag({})'.format(' ⊗ '.join([str(i) for i in reversed(self.state['A'].shape)]))


def run_tests():

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised = kronecker.kron_base.generate_test_data()

    kronecker.kron_base.run_assertions(X, P, kp_literal, kp_optimised)
    kronecker.kron_base.run_assertions(X, P, ks_literal, ks_optimised)
    kronecker.kron_base.run_assertions(X, P, kd_literal, kd_optimised)

    assert np.allclose(np.linalg.inv(kp_literal), kp_optimised.inv().to_array())
    assert np.allclose(np.linalg.inv(kd_literal), kd_optimised.inv().to_array())

    assert np.allclose(kp_literal ** 2, (kp_optimised ** 2).to_array())
    assert np.allclose(kd_literal ** 2, (kd_optimised ** 2).to_array())

    assert np.allclose(kp_literal * kp_literal, (kp_optimised * kp_optimised).to_array())

    print('kron_operators.py: All tests passed')


if __name__ == '__main__':

    run_tests()
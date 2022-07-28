from numpy import ndarray
import kronecker.kron_base

"""
The classes in this file are used to create composite operators. This is the result of adding or multiplying 
two simpler operators together. These classes never need to be created explicityly, but are implicitly 
created whenever two operators are summed or multiplied. 

E.g.

>>> A = KroneckerProduct(A1, A2, A3)
>>> B = KroneckerSum(B1, B2, B3)

>>> C1 = A + B
>>> assert isinstance(C1, OperatorSum)

>>> C2 = A @ B
>>> assert isinstance(C1, OperatorProduct)

This abstraction can be used indefinitely to create higher and higher order composite operators. 
"""



class OperatorSum(kronecker.kron_base.KroneckerOperator):
    """
    Used to represent a chain of Kronecker objects summed together. No need for this class to be
    instatiated by the user. It is used mainly as an internal representation for defining the
    behaviour of composite operators. The internal state of this operator is simply:

    state = {'A': A, 'B': B}

    """

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    def inv(self):
        raise NotImplementedError

    def __init__(self, A: kronecker.kron_base.KroneckerOperator, B: kronecker.kron_base.KroneckerOperator):
        """
        Create an OperatorSum: C = A + B
        """

        kronecker.kron_base.check_operators_consistent(A, B)
        self.state = {'A': A, 'B': B}
        self.shape = self.state['A'].shape

    def __copy__(self):
        new = OperatorSum(self.state['A'].__copy__(), self.state['B'].__copy__())
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = OperatorSum(self.state['A'].__deepcopy__(), self.state['B'].__deepcopy__())
        new.factor = self.factor
        return new

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * (self.state['A'].operate(other) + self.state['B'].operate(other))

    @property
    def T(self):
        return self.factor * OperatorSum(self.state['A'].T, self.state['B'].T)

    def conj(self):
        return self.factor * OperatorSum(self.state['A'].conj(), self.state['B'].conj())

    def to_array(self) -> ndarray:
        return self.factor * (self.state['A'].to_array() + self.state['B'].to_array())

    def __repr__(self):
        return 'OperatorSum({}, {})'.format(self.state['A'].__repr__(), self.state['B'].__repr__())



class OperatorProduct(kronecker.kron_base.KroneckerOperator):
    """
    Used to represent a chain of Kronecker objects matrix-multiplied together. No need for this class to be
    instatiated by the user. It is used mainly as an internal representation for defining the
    behaviour of composite operators.
    """

    def __init__(self, A: kronecker.kron_base.KroneckerOperator, B: kronecker.kron_base.KroneckerOperator):
        """
        Create an OperatorProduct: C = A @ B
        """

        kronecker.kron_base.check_operators_consistent(A, B)
        self.state = {'A': A, 'B': B}
        self.shape = self.state['A'].shape

    def __copy__(self):
        new = OperatorProduct(self.state['A'].__copy__(), self.state['B'].__copy__())
        new.factor = self.factor
        return new

    def __deepcopy__(self, memodict={}):
        new = OperatorProduct(self.state['A'].__deepcopy__(), self.state['B'].__deepcopy__())
        new.factor = self.factor
        return new

    def operate(self, other: ndarray) -> ndarray:
        return self.factor * (self.state['A'] @ (self.state['B'] @ other))

    def inv(self):
        return (1 / self.factor) * OperatorProduct(self.state['B'].inv(), self.state['A'].inv())

    @property
    def T(self):
        return self.factor * OperatorProduct(self.state['B'].T, self.state['A'].T)

    def conj(self):
        return self.factor * OperatorProduct(self.state['B'].conj(), self.state['A'].conj())

    def to_array(self) -> ndarray:
        return self.factor * self.state['A'].to_array() @ self.state['B'].to_array()

    def __repr__(self):
        return 'OperatorProduct({}, {})'.format(self.state['A'].__repr__(), self.state['B'].__repr__())



def run_tests():

    import numpy as np

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    X, Y, P, kp_literal, ks_literal, kd_literal, kp_optimised, ks_optimised, kd_optimised = kronecker.kron_base.generate_test_data()

    def test_sum():

        literal1 = kp_literal + ks_literal
        literal2 = kd_literal - ks_literal / 2
        literal3 = 2.5 * kp_literal + ks_literal / 3 + kd_literal

        optimised1 = kp_optimised + ks_optimised
        optimised2 = kd_optimised - ks_optimised / 2
        optimised3 = 2.5 * kp_optimised + ks_optimised / 3 + kd_optimised

        kronecker.kron_base.run_assertions(X, P, literal1, optimised1)
        kronecker.kron_base.run_assertions(X, P, literal2, optimised2)
        kronecker.kron_base.run_assertions(X, P, literal3, optimised3)

    def test_product():

        literal1 = kp_literal @ ks_literal
        literal2 = kd_literal @ ks_literal / 2
        literal3 = 2.5 * kp_literal @ ks_literal / 3 + kd_literal

        optimised1 = kp_optimised @ ks_optimised
        optimised2 = kd_optimised @ ks_optimised / 2
        optimised3 = 2.5 * kp_optimised @ ks_optimised / 3 + kd_optimised

        kronecker.kron_base.run_assertions(X, P, literal1, optimised1)
        kronecker.kron_base.run_assertions(X, P, literal2, optimised2)
        kronecker.kron_base.run_assertions(X, P, literal3, optimised3)

    test_sum()
    test_product()

    print('kron_composite.py: All tests passed')


if __name__ == '__main__':

    run_tests()
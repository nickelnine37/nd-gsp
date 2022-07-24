import numpy as np
from numpy import ndarray
from typing import Union
from numpy.linalg import inv

from utils.linalg import vec, multiply_tensor_product, multiply_tensor_sum, ten, kronecker_product_literal, kronecker_sum_literal, kronecker_diag_literal
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


def check_valid_matrices(*As) -> bool:
    assert all(isinstance(A, (ndarray, spmatrix)) for A in As)
    assert all(A.ndim == 2 for A in As)
    assert all(A.shape[0] == A.shape[1] for A in As)
    return True


def check_operators_consistent(*operators) -> bool:
    assert all(isinstance(A, KroneckerOperator) for A in operators), f'All operators in this chain must be consistent, but they have types {[type(operator) for operator in operators]} respectively'
    assert all(op1.shape == op2.shape for op1, op2 in zip(operators[1:], operators[:-1])), f'All operators in this chain should have the same shape, but they have shapes {[operator.shape for operator in operators]} respectively'
    return True

def check_blocks_consistent(blocks: list):

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


class KroneckerOperator:
    """
    Base class defining the behaviour of Kronecker-type operators
    """

    __array_priority__ = 10     # increase priority of class, so it takes precedence when mixing matrix multiplications with ndarrays
    factor: float = 1.0         # a scalar factor multiplying the whole operator

    # subclasses need to give values for the following attributes
    shape: tuple = None         # full (N, N) operator shape
    shapes: tuple = None        # (N1, N2, ...) individual shapes
    ndim: int = None            # number of dimensions

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

        if not isinstance(other, KroneckerOperator):
            raise TypeError('Konecker operators can only be added to other Kronecker operators')

        return _SumChain(self, other)

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

        if isinstance(other, ndarray):
            return self.operate(other)

        elif isinstance(other, KroneckerOperator):
            return _ProductChain(self, other)

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




def _run_tests(seed: int=1):

    np.set_printoptions(precision=3, linewidth=500, threshold=500, suppress=True, edgeitems=5)

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

    Y = np.random.randn(N4, N3, N2, N1)
    D = np.random.randn(N4, N3, N2, N1)
    Q = np.random.randn(N4 * N3 * N2 * N1, K)

    # create actual array structures
    kp_literal = kronecker_product_literal(A1, A2, A3, A4)
    ks_literal = kronecker_sum_literal(A1, A2, A3, A4)
    kd_literal = kronecker_diag_literal(D)
    kb_literal = np.block([[kp_literal, kd_literal], [np.zeros(kp_literal.shape), ks_literal]])
    kbd_literal = np.block([[kp_literal, np.zeros(kp_literal.shape)], [np.zeros(kp_literal.shape), ks_literal]])

    # create lazy computation equivelants
    kp_optimised = KroneckerProduct(A1, A2, A3, A4)
    ks_optimised = KroneckerSum(A1, A2, A3, A4)
    kd_optimised = KroneckerDiag(D)
    kb_optimised = KroneckerBlock([[kp_optimised, kd_optimised], [np.zeros(kp_literal.shape), ks_optimised]])
    kbd_optimised = KroneckerBlockDiag([kp_optimised, ks_optimised])


    def run_regular_assertions(literal: ndarray, optimised: KroneckerOperator):

        X = np.random.randn(N4, N3, N2, N1)
        P = np.random.randn(N4 * N3 * N2 * N1, K)

        # test with @ operator
        assert np.allclose(literal @ vec(X), optimised @ vec(X))                      # test forward matrix multiplication
        assert np.allclose(vec(X) @ literal, vec(X) @ optimised)                      # test backward matrix multiplication
        assert np.isclose(vec(X) @ literal @ vec(X), vec(X) @ optimised @ vec(X))     # test quadratic form

        # test with np.matmul
        assert np.allclose(np.matmul(literal, vec(X)), np.matmul(optimised, vec(X)))
        assert np.allclose(np.matmul(vec(X), literal), np.matmul(vec(X), optimised))
        assert np.isclose(np.matmul(vec(X), np.matmul(literal, vec(X))), np.matmul(vec(X), np.matmul(optimised, vec(X))))

        # test summing operation
        assert np.allclose(literal.sum(0), optimised.sum(0))
        assert np.allclose(literal.sum(1), optimised.sum(1))
        assert np.isclose(literal.sum(), optimised.sum())

        # test literal conversion
        assert np.allclose(literal, optimised.to_array())

        # test multiplication onto a data matrix
        assert np.allclose(literal @ P, optimised @ P)
        assert np.allclose(P.T @ literal, P.T @ optimised)
        assert np.allclose(P.T @ literal @ P, P.T @ optimised @ P)

    def run_block_assertions(literal: ndarray, optimised: Union[KroneckerBlock, KroneckerBlockDiag]):

        X1 = np.random.randn(N4 * N3 * N2 * N1 * 2)
        P = np.random.randn(N4 * N3 * N2 * N1 * 2, K)

        # test with @ operator
        assert np.allclose(literal @ X1, optimised @ X1)
        assert np.allclose(X1 @ literal, X1 @ optimised)
        assert np.isclose(X1 @ literal @ X1, X1 @ optimised @ X1)

        assert np.allclose(literal, optimised.to_array())

        # test multiplication onto a data matrix
        assert np.allclose(literal @ P, optimised @ P)
        assert np.allclose(P.T @ literal, P.T @ optimised)
        assert np.allclose(P.T @ literal @ P, P.T @ optimised @ P)


    def test_kronecker_product():
        run_regular_assertions(kp_literal, kp_optimised)

    def test_kronecker_sum():
        run_regular_assertions(ks_literal, ks_optimised)

    def test_kronecker_diag():
        run_regular_assertions(kd_literal, kd_optimised)

    def test_kronecker_block():

        run_block_assertions(kb_literal, kb_optimised)
        run_block_assertions(kb_literal.T, kb_optimised.T)
        run_block_assertions(kbd_literal, kbd_optimised)
        run_block_assertions(kbd_literal.T, kbd_optimised.T)


    def test_product_chain():

        literal = ks_literal @  kd_literal @ kp_literal
        optimised = _ProductChain(ks_optimised, kd_optimised, kp_optimised)
        run_regular_assertions(literal, optimised)

    def test_sum_chain():

        literal = ks_literal + kd_literal + kp_literal
        optimised = _SumChain(ks_optimised, kd_optimised, kp_optimised)
        run_regular_assertions(literal, optimised)

    def test_operator_multiplcation():

        literal = ks_literal @ kd_literal @ kp_literal
        optimised = ks_optimised @ kd_optimised @ kp_optimised
        run_regular_assertions(literal, optimised)

    def test_operator_addition():

        literal = ks_literal + kd_literal + kp_literal
        optimised = ks_optimised + kd_optimised + kp_optimised
        run_regular_assertions(literal, optimised)

    def test_operator_subtraction():

        literal = kd_literal - kp_literal
        optimised = kd_optimised - kp_optimised
        run_regular_assertions(literal, optimised)

    def test_operator_scaling():

        literal = 2 * ks_literal + kd_literal / 5 - (13 / 11) * kp_literal
        optimised = 2 * ks_optimised + kd_optimised / 5 - (13 / 11) * kp_optimised
        run_regular_assertions(literal, optimised)

    def test_kronecker_hadamard():

        literal = 2 * kp_literal * 4 * kp_literal.T
        optimised = 2 * kp_optimised * 4 * kp_optimised.T
        run_regular_assertions(literal, optimised)

    def test_kronecker_pow():

        literal = kp_literal ** 2 + kd_literal ** 3
        optimised = kp_optimised ** 2 + kd_optimised ** 3
        run_regular_assertions(literal, optimised)

    def test_assorted_expressions():

        literal1 = (2 * kp_literal.T - ks_literal) @ kp_literal / 2.2
        literal2 =  -5 * kd_literal @ ks_literal.T + kp_literal @ kp_literal.T
        literal3 = -kp_literal.T @ ks_literal @ kp_literal
        literal4 = (kp_literal - ks_literal @  kd_literal).T @ ks_literal
        literal5 = (kp_literal * kp_literal.T) @  kp_literal.T

        optimised1 = (2 * kp_optimised.T - ks_optimised) @ kp_optimised / 2.2
        optimised2 =  -5 * kd_optimised @ ks_optimised.T + kp_optimised @ kp_optimised.T
        optimised3 = -kp_optimised.T @ ks_optimised @ kp_optimised
        optimised4 = (kp_optimised - ks_optimised @   kd_optimised).T @ ks_optimised
        optimised5 = (kp_optimised * kp_optimised.T) @  kp_optimised.T

        run_regular_assertions(literal1, optimised1)
        run_regular_assertions(literal2, optimised2)
        run_regular_assertions(literal3, optimised3)
        run_regular_assertions(literal4, optimised4)
        run_regular_assertions(literal5, optimised5)

    test_kronecker_product()
    test_kronecker_sum()
    test_kronecker_diag()
    test_kronecker_block()
    test_product_chain()
    test_sum_chain()
    test_operator_multiplcation()
    test_operator_addition()
    test_operator_subtraction()
    test_operator_scaling()
    test_kronecker_hadamard()
    test_kronecker_pow()
    test_assorted_expressions()

    print('All tests passed')


if __name__ == '__main__':

    _run_tests(seed=0)


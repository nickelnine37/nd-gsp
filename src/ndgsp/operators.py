import numpy as np
import jax.numpy as jnp
from pykronecker.base import KroneckerOperator
from pykronecker import KroneckerDiag
from scipy.linalg import block_diag


class KroneckerMuBlock(KroneckerOperator):
    """
    Custom operator to represent blocks of outer products
    """
    
    def __init__(self, Mu: np.ndarray):
            
        self.Mu = Mu
        self.tensor_shape = Mu.shape
        N = int(np.prod(self.tensor_shape))
        self.shape = (N, N)
        self.dtype = Mu.dtype
        
    def operate(self, other: np.ndarray) -> np.ndarray:
        
        # hitting a tensor
        if other.shape == self.tensor_shape:    
            return self.factor * (self.Mu * other).sum(-1)[..., None] * self.Mu
        
        # hitting a vector
        elif other.shape == (self.shape[0], ):
            return self.factor * ((self.Mu * other.reshape(self.tensor_shape)).sum(-1)[..., None] * self.Mu).reshape(-1)
        
        # hitting a matrix of vector columns
        elif (other.shape[0] == self.shape[0]) and other.ndim == 2:
            return self.factor * jnp.stack([((self.Mu * other[:, i].reshape(self.tensor_shape)).sum(-1)[..., None] * self.Mu).reshape(-1) for i in range(other.shape[1])], axis=-1)
        
        else:
            raise ValueError(f'Incompatible dimensions: other has shape {other.shape} which is incompatible with {self.shape}')
        
    @property
    def T(self) -> 'KroneckerMuBlock':
        return self

    def __copy__(self):
        new = self.__class__(self.Mu)
        new.factor = self.factor
        return new
    
    def __deepcopy__(self):
        new = self.__class__(self.Mu.copy())
        new.factor = self.factor
        return new
    
    def __pow__(self):
        pass
    
    def to_array(self) -> np.ndarray:
        Mu_ = self.Mu.reshape(-1, self.Mu.shape[-1])
        return self.factor * block_diag([np.outer(Mu_[i, :], Mu_[i, :]) for i in range(Mu_.shape[0])])

    def __repr__(self) -> str:
        return 'KroneckerMuBlock({})'.format(' ⊗ '.join([str(Ni) for Ni in self.tensor_shape]))

    def __str__(self) -> str:
        return 'KroneckerMuBlock({})'.format(' ⊗ '.join([str(Ni) for Ni in self.tensor_shape]))
    
    def conj(self):
        pass
    
    def diag(self):
        return self.factor * KroneckerDiag(self.Mu ** 2)
    
    def inv(self):
        pass
    
        
class KroneckerExpanded(KroneckerOperator):
    """
    Custom operator to represent A ⊗ I efficiently
    """
    
    def __init__(self, Op: KroneckerOperator, C: int):
            
        self.Op = Op
        self.C = C
        self.tensor_shape = tuple(list(Op.tensor_shape) + [C])
        N = int(np.prod(self.tensor_shape))
        self.shape = (N, N)
        self.dtype = Op.dtype
        
    def operate(self, other: np.ndarray) -> np.ndarray:
        
        # hitting a tensor
        if other.shape == self.tensor_shape:   
            return self.factor * jnp.stack([self.Op.operate(other[..., i]) for i in range(self.C)], axis=-1)
        
        # hitting a vector
        elif other.shape == (self.shape[0], ):
            other_ = other.reshape(self.tensor_shape)
            return self.factor * jnp.stack([self.Op.operate(other_[..., i]) for i in range(self.C)], axis=-1).reshape(-1)
        
        # hitting a matrix of vector columns
        elif (other.shape[0] == self.shape[0]) and other.ndim == 2:
            out = []
            for j in range(other.shape[1]):
                other_ = other[:, j].reshape(self.tensor_shape)
                out.append(jnp.stack([self.Op.operate(other_[..., i]) for i in range(self.C)], axis=-1).reshape(-1))
                
            return self.factor * jnp.stack(out, axis=-1)
        else:
            raise ValueError(f'Incompatible dimensions: other has shape {other.shape} which is incompatible with {self.shape}')
            
    @property
    def T(self) -> 'KroneckerExpanded':
        new = self.__class__(self.Op.T, self.C)
        new.factor = self.factor
        return new
    
    def __copy__(self):
        new = self.__class__(self.Op.__copy__(), self.C)
        new.factor = self.factor
        return new
    
    def __deepcopy__(self):
        new = self.__class__(self.Op.__deepcopy__(), self.C)
        new.factor = self.factor
        return new
    
    def __pow__(self, other: float):
        return self.factor ** other * KroneckerExpanded(self.Op.__pow__(other), self.C)
    
    def __repr__(self) -> str:
        return self.Op.__repr__() + f' ⊗ I_{self.C}'

    def __str__(self) -> str:
        return self.Op.__repr__() + f' ⊗ I_{self.C}'
    
    def conj(self):
        return np.conj(self.factor) * KroneckerExpanded(self.Op.conj(), self.C)
    
    def diag(self):
        return self.factor * KroneckerExpanded(self.Op.diag(), self.C)
    
    def inv(self):
        return self.factor ** -1 * KroneckerExpanded(self.Op.inv(), self.C)
    
    def to_array(self):
        return self.factor * np.kron(self.Op.to_array(), np.eye(self.C))



class KroneckerQXBlock(KroneckerOperator):
    """
    Special non-square operator to represent the off-diagonal blocks in the preconditioned 
    coefficient matrix Q
    """
    
    def __init__(self, OpLeft: KroneckerOperator, X: np.ndarray, UMDM: np.ndarray):
            
        self.OpLeft = OpLeft
        self.X = X
        self.UMDM = UMDM
        
        self.N, self.M = X.shape
        self.C = UMDM.shape[0] // self.M
                
        self.tensor_shape = None
        self.shape = (self.N * self.C, self.M * self.C)
        self.dtype = OpLeft.dtype
        
        self.transpose = False
        
    def operate(self, other: np.ndarray) -> np.ndarray:
        
        # hitting vector or length NC
        if self.transpose:
            return self.factor * self.UMDM.T @ (self.X.T @ (self.OpLeft.T @ other).reshape(-1, self.C)).ravel()
        
        # hitting vector of length MC
        else:    
            return self.factor * self.OpLeft @ (self.X @ (self.UMDM @ other).reshape(self.M, -1)).ravel()
        
    @property
    def T(self) -> 'KroneckerQXBlock':
        new = KroneckerQXBlock(self.OpLeft, self.X, self.UMDM)
        new.factor = self.factor
        new.transpose = ~self.transpose
        new.shape = (self.shape[1], self.shape[0])
        return new

    def __copy__(self):
        new = self.__class__(self.OpLeft, self.X, self.UMDM)
        new.factor = self.factor
        return new
    
    def __deepcopy__(self):
        new = self.__class__(self.OpLeft.copy(), self.X.copy(), self.UMDM.copy())
        new.factor = self.factor
        return new
    
    def __pow__(self):
        pass
    
    def __repr__(self):
        return 'KroneckerQXBlock'
    
    def conj(self):
        pass
    
    def diag(self):
        pass
    
    def inv(self):
        pass
    
    def to_array(self):
        pass
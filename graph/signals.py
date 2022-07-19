from numpy import ndarray
import numpy as np
# import pandas as pd
# from typing import Union


class PartiallyObservedGraphSignal:
    """
    A partially observed graph signal: this is a length-N vector combined with a binary length-N
    sensing vector, indicating where observations were made. Where no observation was made, i.e.
    s[i] == 0, we shold have that y[i] == 0 too.
    """

    def __init__(self, y: ndarray, s: ndarray):

        assert isinstance(y, ndarray), 'y should be an ndarray'
        assert isinstance(s, ndarray), 's should be an ndarray'
        assert y.ndim == 1, 'y should be 1-dimensional'
        assert s.ndim == 1, 's should be 1-dimensional'
        assert len(y) == len(s), f'y and s should b the same size but they have shape {len(y)} and {len(s)} respectively'

        self.y = y
        self.s = s.astype(bool)
        assert np.isclose(y[~s].sum(), 0), 'there should be zeros in y where no observations where made'

        self.N = len(y)
        self.N_ = self.s.sum()     # the number of observations made

    def down_project_signal(self, f: ndarray):
        """
        Condense f such that it contains only elements specified by s
        f (N, ) -> f_ (N_, )
        """
        assert len(f) == self.N
        return f[self.s]

    def up_project_signal(self, f_: ndarray):
        """
        Upsample f so that it is padded with zeros where no observation was made
        f_ (N_, ) -> f (N, )
        """
        assert len(f_) == self.N_
        f = np.zeros(self.N)
        f[self.s] = f_
        return f

    def down_project_operator(self, A: ndarray):
        """
        Condense a matrix A, removing rows and columns as specified by s
        A (N, N) -> A_ (N_, N_)
        """
        assert A.shape == (self.N, self.N)
        return A[:, self.s][self.s, :]

    def up_project_operator(self, A_: ndarray):
        """
        Upsample a matrix A_, adding zeros columns and rows appropriately
        A_ (N_, N_) -> A (N, N)
        """
        assert A_.shape == (self.N_, self.N_)
        A = np.zeros(self.N ** 2)
        A[(self.s[:, None] * self.s[None, :]).reshape(-1)] = A_.reshape(-1)
        return A.reshape(self.N, self.N)






class PartiallyObservedProductGraphSignal:


    def __init__(self, Y: ndarray, S: ndarray):

        assert isinstance(Y, ndarray), 'Y should be an ndarray'
        assert isinstance(S, ndarray), 'S should be an ndarray'
        assert Y.ndim == 2, 'Y should be 2-dimensional'
        assert S.ndim == 2, 'S should be 2-dimensional'
        assert Y.shape == S.shape, f'Y and S should have the same shape but they have shape {Y.shape} and {S.shape} respectively'

        self.Y = Y
        self.S = S.astype(bool)
        self.N, self.T = Y.shape


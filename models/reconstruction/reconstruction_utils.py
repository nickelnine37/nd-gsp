import numpy as np
from numpy import ndarray


def get_y_and_s(signal: ndarray):
    """
    Take in a user-provided partially observed graph signal. This can be either:

        * an array containing nans to indicate missing data,
        * an np.ma.MaskedArray object

    Return this array with the missing values filled with zeros, and a boolean array
    of the same shape holding True where observations were made.

    """
    if isinstance(signal, np.ma.MaskedArray):
        s = ~signal.mask.copy()
        y = signal.data.copy()

    elif isinstance(signal, ndarray):
        s = ~np.isnan(signal)
        y = signal.copy()

    else:
        raise TypeError('signal should be an array or a masked array')

    y[~s] = 0

    return y, s.astype(float)


class SignalProjector:

    def __init__(self, signal: ndarray):
        """
        s is a boolean array specifying where measurements were made
        """

        _, s = get_y_and_s(signal)

        self.s = s.astype(bool)
        self.N = len(s)
        self.N_ = int(s.sum())

    def down_project_signal(self, f: ndarray):
        """
        Condense a vector f such that it contains only elements specified by s
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
        assert A.shape == (self.N, self.N), f'passed array should have shape {(self.N, self.N)} but it has shape {A.shape}'
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

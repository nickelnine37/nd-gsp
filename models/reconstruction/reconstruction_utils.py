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

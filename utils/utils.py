import numpy as np
from numpy import ndarray, diag
from scipy.sparse import csr_array, spmatrix, triu as sptriu, tril as sptril
from typing import Union








def isnumeric(obj) -> bool:
    """
    Check if an object is a numeric type, i.e. float, int, ndarray etc.
    """

    try:
        obj + 0
        return True

    except TypeError:
        return False#



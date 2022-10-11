from __future__ import annotations
import numpy as np
from numpy import ndarray, diag
from typing import List, Union
import jax.numpy as jnp
from pykronecker.base import KroneckerOperator

Numeric = Union[int, float, complex, np.number]
Array = Union[ndarray, jnp.ndarray]
Operator = Union[Array, KroneckerOperator]



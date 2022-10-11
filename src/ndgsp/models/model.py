from abc import ABC, abstractmethod
from typing import Tuple

from ndgsp.graph.graphs import BaseGraph, ProductGraph
from ndgsp.utils.types import Signal
import jax.numpy as jnp
import pandas as pd
import numpy as np


class Model(ABC):

    @staticmethod
    def get_Y_and_S(signal: Signal) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Convert a graph signal containing nans into two separate arrays. The first is a copy of signal, with nans
        replaced by 0. The second is a binary array containing zeros where nans were, and ones elsewhere
        """

        assert isinstance(signal, Signal), f'signal should be either a numpy array, jax array, or pandas DataFrame, but it is a {type(signal)}'
        assert signal.ndim > 1, f'signal should have more than 1 dimension, but it has {signal.ndim}'

        if isinstance(signal, pd.DataFrame):
            Y = jnp.asarray(signal.values.astype(float))

        elif isinstance(signal, np.ndarray):
            Y = jnp.asarray(signal.astype(float))

        else:
            Y = signal.astype(float)

        S_ = jnp.isnan(Y)

        return Y.at[S_].set(0), (~S_).astype(float)

    @staticmethod
    def check_consistent(signal: Signal, graph: ProductGraph):
        assert signal.ndim == graph.ndim, f'The signal and graph should have the same number of dimensions, but they have {signal.ndim} and {graph.ndim} respectively'
        assert signal.shape == graph.A.tensor_shape, f'The signal and graph should have consistent shapes, but they have {signal.shape} and {graph.A.tensor_shape} respectively'



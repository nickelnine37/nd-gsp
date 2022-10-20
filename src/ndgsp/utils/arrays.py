from ndgsp.utils.types import Array
import jax.numpy as jnp


def expand_dims(x: Array, where='left', n: int = 1):
    """
    Add n new axes to the left or right of an array
    """

    assert where in ['left', 'right'], f'where should be "left" or "right" but it is {where}'

    for i in range(n):

        if where == 'left':
            x = jnp.expand_dims(x, axis=0)

        else:
            x = jnp.expand_dims(x, axis=-1)

    return x


def outer_product(x, y):
    """
    A generalised version of the outer product
    """

    x = x.squeeze()
    y = y.squeeze()

    return expand_dims(x, 'right', y.ndim) * expand_dims(y, 'left', x.ndim)

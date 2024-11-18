import jax.numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Domain



def create_src_field(N, x, y, domain, omega):
    """
    Create a source field with bilinear interpolation at the given (x, y) coordinates
    and compute its Fourier Series.

    Parameters
    ----------
    N : tuple of int
        The size of the source field array as (height, width).
    x : float
        The x-coordinate of the source (can be a fractional value).
    y : float
        The y-coordinate of the source (can be a fractional value).
    domain : array-like
        The domain for the Fourier Series calculation.
    omega : float
        The scaling factor for the Fourier Series.

    Returns
    -------
    array
        The computed Fourier Series of the source field.

    Notes
    -----
    - The source field uses bilinear interpolation to assign values at the 
      four nearest grid points.
    - The Fourier Series is computed using the `FourierSeries` function 
      (assumed to be defined elsewhere).

    Example
    -------
    >>> import jax.numpy as jnp
    >>> N = (100, 100)
    >>> x, y = 45.5, 76.3
    >>> domain = jnp.linspace(0, 1, 100)
    >>> omega = 2.0
    >>> src = create_src_field(N, x, y, domain, omega)
    >>> src.shape
    (100,)

    """

    # check that the domain is a jwave Domain object
    if not isinstance(domain, Domain):
        raise ValueError("The domain must be a jwave Domain object")

    src_field = jnp.zeros(N).astype(jnp.complex64)

    # Ensure the coordinates are within the valid range
    x = jnp.clip(x, 0, N[0] - 1)
    y = jnp.clip(y, 0, N[1] - 1)

    # Compute the floor and ceil of the coordinates
    x_floor = jnp.floor(x).astype(jnp.int32)
    y_floor = jnp.floor(y).astype(jnp.int32)
    x_ceil = jnp.ceil(x).astype(jnp.int32)
    y_ceil = jnp.ceil(y).astype(jnp.int32)

    # Compute interpolation weights
    w_floor_floor = (x_ceil - x) * (y_ceil - y)
    w_floor_ceil = (x_ceil - x) * (y - y_floor)
    w_ceil_floor = (x - x_floor) * (y_ceil - y)
    w_ceil_ceil = (x - x_floor) * (y - y_floor)

    # Apply weights to the four surrounding grid points
    src_field = src_field.at[x_floor, y_floor].add(w_floor_floor)
    src_field = src_field.at[x_floor, y_ceil].add(w_floor_ceil)
    src_field = src_field.at[x_ceil, y_floor].add(w_ceil_floor)
    src_field = src_field.at[x_ceil, y_ceil].add(w_ceil_ceil)

    # Compute the Fourier Series of the source field
    src = FourierSeries(jnp.expand_dims(src_field, -1), domain) * omega

    return src

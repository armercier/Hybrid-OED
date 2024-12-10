import jax.numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Domain
from scipy.special import hankel1
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium
from jax import jit




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

@jit
def solve_helmholtz_general(sound_speed: jnp.ndarray, domain: Domain, src_coord: tuple, omega: float, pml_size: int = 10):
    """
    Solve the general Helmholtz equation for a given sound speed distribution, source location, and frequency.

    This function computes the acoustic field by solving the Helmholtz equation in a medium
    characterized by a given sound speed. It creates a source field, defines the medium properties,
    and solves the Helmholtz equation using a specified solver.

    Parameters
    ----------
    sound_speed : numpy.ndarray
        A 2D array representing the spatial distribution of sound speed in the medium (in m/s).
    src_coord : tuple
        A tuple (x, y) specifying the source location in the domain (indices).
    omega : float
        Angular frequency of the wave (in rad/s).

    Returns
    -------
    numpy.ndarray
        A 2D array representing the acoustic field on the grid, with dimensions matching the sound speed array.

    Notes
    -----
    - The `create_src_field` function is used to generate the source field based on the source coordinates.
    - The medium is modeled with a density of 1000 kg/m³ and includes Perfectly Matched Layer (PML) boundaries.
    - The Helmholtz equation is solved using the `helmholtz_solver` function.

    Examples
    --------
    >>> import numpy as np
    >>> sound_speed = np.ones((100, 100)) * 1500  # Speed of sound in water
    >>> src_coord = (50, 50)  # Source in the center
    >>> omega = 2 * np.pi * 1e3  # 1 kHz frequency
    >>> field = solve_helmholtz_general(sound_speed, src_coord, omega)
    >>> field.shape
    (100, 100)
    """
    N = sound_speed.shape

    src = create_src_field(N, src_coord[0], src_coord[1], domain, omega)
    medium = Medium(
        domain=domain, 
        sound_speed=sound_speed, 
        pml_size=pml_size
    )
    field = helmholtz_solver(medium, omega, src)
    return field.on_grid.squeeze()


def get_analytical_greens_function(L_x, L_y, acoustic_velocity, x_source, y_source, f):
    """
    Compute the Green's function for a 2D wave equation.

    This function calculates the analytical Green's function for a 2D wave equation given
    the dimensions of the domain, source location, and frequency. The function 
    uses the Hankel function of the first kind to compute the Green's function 
    in two dimensions.

    Parameters
    ----------
    L_x : int
        Length of the domain in the x-direction (number of grid points) in meters.
    L_y : int
        Length of the domain in the y-direction (number of grid points) in meters.
    acoustic_velocity : float
        The velocity of the medium (in m/s).
    x_source : float
        x-coordinate of the source point.
    y_source : float
        y-coordinate of the source point.
    f : float
        Frequency of the wave (in Hz).

    Returns
    -------
    jax.numpy.ndarray
        A 2D array representing the Green's function values at each grid point.

    Notes
    -----
    - The Green's function in 2D is computed using the formula:
      `G = (1j / 4) * Hankel1(0, K * R)`
      where `K` is the wave number, `R` is the distance between observation
      and source points, and `Hankel1` is the Hankel function of the first kind.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> L_x, L_y = 100, 100
    >>> acoustic_velocity = 1500.0
    >>> x_source, y_source = 50.0, 50.0
    >>> f = 1e3
    >>> G = get_analytical_greens_function(L_x, L_y, acoustic_velocity, x_source, y_source, f)
    >>> G.shape
    (100, 100)
    """
    # Define the frequency
    omega = 2 * jnp.pi * f  # angular frequency

    # Define the grid of observation points
    x = jnp.linspace(0, L_x, L_x)
    y = jnp.linspace(0, L_y, L_y)
    X, Y = jnp.meshgrid(x, y)

    # Define the source location
    x_prime = jnp.full((L_x,), x_source)
    y_prime = jnp.full((L_y,), L_y - y_source)
    X_prime, Y_prime = jnp.meshgrid(x_prime, y_prime)

    # Calculate the distance between observation points and source points
    R = jnp.sqrt((X - X_prime) ** 2 + (Y - Y_prime) ** 2)
    K = omega / acoustic_velocity  # wave number

    # Calculate the Green's function 2D
    A = 1j / 4
    G = A * hankel1(0, K * R)
    return G

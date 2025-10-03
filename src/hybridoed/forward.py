import jax.numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Domain
from scipy.special import hankel1
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium
import jax
from jax import jit, lax
from typing import Sequence, Tuple





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

def generate_2D_gridded_src_rec_positions(N=(70, 70), num_sources=5, num_receivers=5):
    """
    Generate a 2D grid of positions for sources and receivers with more sources than receivers,
    arranged in a staggered grid. The offset between a source and a receiver is half the source spacing.

    Parameters:
    - N: Tuple[int, int], dimensions of the 2D grid (Nx, Ny).
    - num_sources: int, number of source positions along each axis.
    - num_receivers: int, number of receiver positions along each axis.

    Returns:
    - src_coords: jnp.ndarray, source positions as a 2D array.
    - recv_coords: jnp.ndarray, receiver positions as a 2D array.
    """
    Nx, Ny = N

    # Generate evenly spaced indices for sources
    src_x = jnp.linspace(5, Nx - 5, num_sources, dtype=jnp.float32)
    src_y = jnp.linspace(5, Ny - 5, num_sources, dtype=jnp.float32)

    # Compute receiver positions with fewer points
    recv_x = jnp.linspace(5 + (src_x[1] - src_x[0]) / 2 + 0.1, Nx - 5 - (src_x[1] - src_x[0]) / 2 + 0.1, num_receivers, dtype=jnp.float32)
    recv_y = jnp.linspace(5 + (src_y[1] - src_y[0]) / 2 + 0.1, Ny - 5 - (src_y[1] - src_y[0]) / 2 + 0.1, num_receivers, dtype=jnp.float32)

    # Create 2D grid coordinates for sources and receivers
    src_coords = jnp.array([[x + 0.1, y + 0.1] for x in src_x for y in src_y], dtype=jnp.int32)
    recv_coords = jnp.array([[x + 0.1, y + 0.1] for x in recv_x for y in recv_y], dtype=jnp.int32)

    return src_coords, recv_coords

def generate_src_rec_positions(
    src_x: Sequence[float],
    src_y: Sequence[float],
    rec_x: Sequence[float],
    rec_y: Sequence[float],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Given explicit source‐grid and receiver‐grid coords, return flattened
    (N,2) arrays of (i,j) positions.

    Parameters
    ----------
    src_x, src_y : sequences of floats
        1D arrays of X‐ and Y‐coordinates for sources.
    rec_x, rec_y : sequences of floats
        1D arrays of X‐ and Y‐coordinates for receivers.

    Returns
    -------
    src_coords, recv_coords : jnp.ndarray, shape (len(src_x)*len(src_y), 2), dtype float32
        Source positions as [[x0,y0], [x1,y0], …] in mesh order.
    recv_coords, same shape logic for receivers.
    """
    sx, sy = jnp.meshgrid(jnp.array(src_x, dtype=jnp.float32),
                          jnp.array(src_y, dtype=jnp.float32),
                          indexing="xy")
    rx, ry = jnp.meshgrid(jnp.array(rec_x, dtype=jnp.float32),
                          jnp.array(rec_y, dtype=jnp.float32),
                          indexing="xy")

    src_coords  = jnp.stack([sx.ravel(),  sy.ravel()],  axis=-1)
    recv_coords = jnp.stack([rx.ravel(),  ry.ravel()], axis=-1)

    return src_coords, recv_coords

def ricker_wavelet(t, f, t_shift=0.0):
    """
    Generate a Ricker wavelet with a time shift.

    Parameters:
        t (array-like): Time axis (e.g., jax.numpy array).
        f (float): Central frequency of the wavelet.
        t_shift (float): Time shift for the wavelet (default is 0.0).

    Returns:
        jax.numpy.ndarray: Ricker wavelet values at the given time points.
    """
    t_shifted = t - t_shift  # Apply the time shift
    pi2 = (jnp.pi ** 2)
    a = (pi2 * f ** 2) * (t_shifted ** 2)
    wavelet = (1 - 2 * a) * jnp.exp(-a)
    
    return wavelet


def acoustic2D(velocity,
               density,
               source_i,
               f0,
               dx, dy, dt,
               n_steps,
               receiver_is=None,
               output_wavefield=True,
               ):
    """Simulate seismic waves through a 2D velocity model"""

    assert density.shape == velocity.shape
    nx, ny = velocity.shape

    # precompute some arrays
    pressure_present = jnp.zeros_like(velocity)
    pressure_past = jnp.zeros_like(velocity)
    kappa = density*(velocity**2)
    density_half_x = jnp.pad(0.5 * (density[1:nx,:]+density[:nx-1,:]), [[0,1],[0,0]], mode="edge")
    density_half_y = jnp.pad(0.5 * (density[:,1:ny]+density[:,:ny-1]), [[0,0],[0,1]], mode="edge")
    t0 = 1.2 / f0
    factor = 1e4
    v_source = velocity[source_i[0], source_i[1]]

    # TODO: add absorbing boundary

    def step(carry, it):
        pressure_past, pressure_present = carry

        t = it*dt

        # compute the first spatial derivatives divided by density
        pressure_x = jnp.pad((pressure_present[1:nx,:]-pressure_present[:nx-1,:]) / dx, [[0,1],[0,0]], mode="constant", constant_values=0.)
        pressure_y = jnp.pad((pressure_present[:,1:ny]-pressure_present[:,:ny-1]) / dy, [[0,0],[0,1]], mode="constant", constant_values=0.)
        pressure_density_x = pressure_x / density_half_x
        pressure_density_y = pressure_y / density_half_y

        # compute the second spatial derivatives
        pressure_xx = jnp.pad((pressure_density_x[1:nx,:]-pressure_density_x[:nx-1,:]) / dx, [[1,0],[0,0]], mode="constant", constant_values=0.)
        pressure_yy = jnp.pad((pressure_density_y[:,1:ny]-pressure_density_y[:,:ny-1]) / dy, [[0,0],[1,0]], mode="constant", constant_values=0.)

        # advance wavefield
        pressure_future = - pressure_past \
                          + 2 * pressure_present \
                          + dt*dt*(pressure_xx+pressure_yy)*kappa

        # add the source
        # Ricker source time function (second derivative of a Gaussian)
        a = (jnp.pi**2)*f0*f0
        source_term = factor * (1 - 2*a*(t-t0)**2)*jnp.exp(-a*(t-t0)**2)
        pressure_future = pressure_future.at[source_i[0], source_i[1]].add(
            dt*dt*(4*jnp.pi*(v_source**2)*source_term))# latest seismicCPML normalisation

        # extract outputs
        y = []
        if receiver_is is not None:
            gather = pressure_future[receiver_is[:,0], receiver_is[:,1]]
            y.append(gather)
        if output_wavefield:
            y.append(pressure_future)

        # move new values to old values (the present becomes the past, the future becomes the present)
        return (pressure_present, pressure_future), y

    _, y = lax.scan(step, (pressure_past, pressure_present), jnp.arange(n_steps))

    return y

def acoustic2D_pml(velocity,
                   density,
                   source_i,
                   f0,
                   dx, dy, dt,
                   n_steps,
                   receiver_is=None,
                   output_wavefield=True,
                   pml_width=20,
                   R_coeff=1e-3):
    """2D acoustic FD with a simple PML/sponge absorbing boundary,
       and the Ricker source scaled per seismicCPML normalization."""
    nx, ny = velocity.shape
    assert density.shape == velocity.shape

    # time levels
    pressure_past    = jnp.zeros_like(velocity)
    pressure_present = jnp.zeros_like(velocity)

    # bulk modulus and half-cell densities
    kappa      = density * (velocity**2)
    rho_x_half = jnp.pad(0.5*(density[1:,:] + density[:-1,:]), [[0,1],[0,0]], mode="edge")
    rho_y_half = jnp.pad(0.5*(density[:,1:] + density[:,:-1]), [[0,0],[0,1]], mode="edge")

    # Ricker source parameters
    t0     = 1.2 / f0
    factor = 1e3
    v_source = velocity[source_i[0], source_i[1]]
    # print(f"v_source: {v_source}")

    # build PML damping profile
    sigma_max = -(3.0 * velocity.max() * jnp.log(R_coeff)) / (2.0 * pml_width * dx)
    def make_sigma(n, npml):
        i = jnp.arange(n)
        left  = jnp.where(i < npml,       (npml - i) / npml, 0.0)
        right = jnp.where(i >= n - npml, (i - (n-npml-1)) / npml, 0.0)
        return sigma_max * (left**2 + right**2)
    σx = make_sigma(nx, pml_width)
    σy = make_sigma(ny, pml_width)
    sigma2d = jnp.add.outer(σx, σy)

    def step(carry, it):
        p_nm1, p_n = carry
        t = it * dt

        # spatial gradients -> laplacian
        dp_dx = jnp.pad((p_n[1:,:] - p_n[:-1,:]) / dx, [[0,1],[0,0]], 'constant')
        dp_dy = jnp.pad((p_n[:,1:] - p_n[:,:-1]) / dy, [[0,0],[0,1]], 'constant')
        dp_dx /= rho_x_half
        dp_dy /= rho_y_half
        d2p_dx2 = jnp.pad((dp_dx[1:,:] - dp_dx[:-1,:]) / dx, [[1,0],[0,0]], 'constant')
        d2p_dy2 = jnp.pad((dp_dy[:,1:] - dp_dy[:,:-1]) / dy, [[0,0],[1,0]], 'constant')

        # Ricker source term
        t_source = t
        a = (jnp.pi**2) * f0**2
        ricker = factor * (1 - 2*a*(t_source - t0)**2) * jnp.exp(-a*(t_source - t0)**2)

        # PML damping coefficient beta
        β = sigma2d * dt / 2.0

        # finite-difference update with damping
        lap = dt*dt * (d2p_dx2 + d2p_dy2) * kappa
        p_np1 = (lap + 2*p_n - (1 - β)*p_nm1) / (1 + β)

        # inject scaled source per seismicCPML
        source_term = dt*dt * (4*jnp.pi*(v_source**2) * ricker)
        p_np1 = p_np1.at[source_i[0], source_i[1]].add(source_term)

        # collect outputs
        out = []
        if receiver_is is not None:
            out.append(p_np1[receiver_is[:,0], receiver_is[:,1]])
        if output_wavefield:
            out.append(p_np1)

        return (p_n, p_np1), out

    (_, _), ys = lax.scan(step,
                          (pressure_past, pressure_present),
                          jnp.arange(n_steps))
    return ys


def acoustic2D_pml_minmem(velocity,
                         density,
                         source_i,
                         f0,
                         dx, dy, dt,
                         n_steps,
                         receiver_is=None,
                         output_wavefield=True,
                         pml_width=20,
                         R_coeff=1e-3):
    """
    Memory‐optimized 2D acoustic FD with PML absorbing boundary.

    This version stores only two time‐levels and uses 1D PML profiles,
    avoiding a full 2D sigma array. It also allows dropping the wavefield
    output to save memory if only receiver traces are needed.
    """
    nx, ny = velocity.shape
    assert density.shape == velocity.shape

    # Allocate only two full‐grid wavefields
    p_nm1 = jnp.zeros((nx, ny), dtype=jnp.float32)
    p_n   = jnp.zeros((nx, ny), dtype=jnp.float32)

    # Bulk modulus and half‐cell densities (float32)
    kappa      = (density * velocity**2).astype(jnp.float32)
    rho_x_half = jnp.pad(0.5*(density[1:,:] + density[:-1,:]), [[0,1],[0,0]], mode="edge").astype(jnp.float32)
    rho_y_half = jnp.pad(0.5*(density[:,1:] + density[:,:-1]), [[0,0],[0,1]], mode="edge").astype(jnp.float32)

    # Precompute constants
    t0      = (jnp.array(1.2) / f0).astype(jnp.float32)
    factor  = jnp.array(1e4, dtype=jnp.float32)
    # v_src   = jnp.array(velocity[source_i[0], source_i[1]], dtype=jnp.float32)
    # compute floor‐indices (clamped so i0+1, j0+1 stay in bounds)
    sx_f, sy_f = source_i            # now floats
    i0 = jnp.clip(jnp.floor(sx_f).astype(jnp.int32), 0, nx-2)
    j0 = jnp.clip(jnp.floor(sy_f).astype(jnp.int32), 0, ny-2)
    di = sx_f - i0
    dj = sy_f - j0

    # fetch the four corner values of velocity
    v00 = velocity[i0,   j0  ]
    v10 = velocity[i0+1, j0  ]
    v01 = velocity[i0,   j0+1]
    v11 = velocity[i0+1, j0+1]

    # bilinear weights
    w00 = (1 - di)*(1 - dj)
    w10 = di      *(1 - dj)
    w01 = (1 - di)*dj
    w11 = di      *dj

    # now interpolated velocity at (sx_f,sy_f)
    v_src = (w00 * v00 +
            w10 * v10 +
            w01 * v01 +
            w11 * v11).astype(jnp.float32)


    a_const = jnp.array(jnp.pi**2 * f0**2).astype(jnp.float32)
    dt2     = jnp.array(dt * dt).astype(jnp.float32)

    # Build 1D PML damping profiles (no 2D array)
    sigma_max = -(3.0 * velocity.max() * jnp.log(R_coeff)) / (2.0 * pml_width * dx)
    i = jnp.arange(nx, dtype=jnp.float32)
    j = jnp.arange(ny, dtype=jnp.float32)
    # Quadratic ramp on edges
    ramp_x = jnp.where(i < pml_width, (pml_width - i) / pml_width, 0.0)
    ramp_x = jnp.where(i >= nx - pml_width, (i - (nx - pml_width - 1)) / pml_width, ramp_x)
    ramp_y = jnp.where(j < pml_width, (pml_width - j) / pml_width, 0.0)
    ramp_y = jnp.where(j >= ny - pml_width, (j - (ny - pml_width - 1)) / pml_width, ramp_y)
    beta_x = (sigma_max * ramp_x**2 * dt / 2.0).astype(jnp.float32)  # shape (nx,)
    beta_y = (sigma_max * ramp_y**2 * dt / 2.0).astype(jnp.float32)  # shape (ny,)

    def step(carry, it):
        p_prev, p_curr = carry
        t = it * dt

        # --- 1) compute the “nominal” FD update (no source) ---
        # spatial gradients
        dp_dx = jnp.pad((p_curr[1:,:] - p_curr[:-1,:]) / dx,
                        [[0,1],[0,0]], 'constant') / rho_x_half
        dp_dy = jnp.pad((p_curr[:,1:] - p_curr[:,:-1]) / dy,
                        [[0,0],[0,1]], 'constant') / rho_y_half

        lap = dt2 * (
            jnp.pad((dp_dx[1:,:] - dp_dx[:-1,:]) / dx, [[1,0],[0,0]], 'edge')
        + jnp.pad((dp_dy[:,1:] - dp_dy[:,:-1]) / dy, [[0,0],[1,0]], 'edge')
        ) * kappa

        β_sum = beta_x[:, None] + beta_y[None, :]

        p_nominal = (lap + 2*p_curr - (1 - β_sum)*p_prev) / (1 + β_sum)

        # --- 2) build the Ricker source injection term ---
        ricker = factor * (1 - 2*a_const*(t - t0)**2) * jnp.exp(-a_const*(t - t0)**2)
        src_term = dt2 * (4*jnp.pi*(v_src**2)) * ricker   # scalar

        # --- 3) bilinear‐inject at FLOAT location (sx_f, sy_f) ---
        sx_f, sy_f = source_i        # now floats
        # floor‐indices, then clip to [0, nx-2]×[0, ny-2]
        i0 = jnp.clip(jnp.floor(sx_f).astype(jnp.int32), 0, nx-2)
        j0 = jnp.clip(jnp.floor(sy_f).astype(jnp.int32), 0, ny-2)
        di = sx_f - i0
        dj = sy_f - j0

        w00 = (1 - di)*(1 - dj)
        w10 = di      *(1 - dj)
        w01 = (1 - di)*dj
        w11 = di      *dj

        # start from the nominal update
        p_next = p_nominal
        # scatter‐add each corner
        p_next = p_next.at[i0,   j0  ].add(w00 * src_term)
        p_next = p_next.at[i0+1, j0  ].add(w10 * src_term)
        p_next = p_next.at[i0,   j0+1].add(w01 * src_term)
        p_next = p_next.at[i0+1, j0+1].add(w11 * src_term)

        # --- 4) gather outputs ---
        # unpack float positions
        sx_rec = receiver_is[:, 0]   # shape (Nrec,)
        sy_rec = receiver_is[:, 1]   # shape (Nrec,)

        # compute integer “corners” and fractional offsets
        i0r = jnp.clip(jnp.floor(sx_rec).astype(jnp.int32), 0, nx-2)
        j0r = jnp.clip(jnp.floor(sy_rec).astype(jnp.int32), 0, ny-2)
        di  = sx_rec - i0r
        dj  = sy_rec - j0r

        # corner values
        v00 = p_next[i0r,   j0r  ]  # shape (Nrec,)
        v10 = p_next[i0r+1, j0r  ]
        v01 = p_next[i0r,   j0r+1]
        v11 = p_next[i0r+1, j0r+1]

        # bilinear weights
        w00 = (1-di)*(1-dj)
        w10 = di     *(1-dj)
        w01 = (1-di)*dj
        w11 = di     *dj

        # interpolated receiver traces
        rec_vals = w00*v00 + w10*v10 + w01*v01 + w11*v11  # shape (Nrec,)

        out = []
        out.append(rec_vals)
        if output_wavefield:
            out.append(p_next)

        return (p_curr, p_next), out

    (_, _), ys = lax.scan(step, (p_nm1, p_n), jnp.arange(n_steps))
    return ys

def _cpml_1d(n, dx, dt, pml_width, c_ref, f0,
             R=1e-6, m=3, kappa_max=3.0, alpha_max=None):
    """
    Build 1-D CPML profiles and ADE coeffs at a given index grid (length n).
    Returns dict with sigma, kappa, alpha, a, b (all shape (n,)).
    Zero in interior; graded within 'pml_width' cells from each boundary.
    """
    if alpha_max is None:
        alpha_max = 2.0 * jnp.pi * f0  # helps low-freq/grazing

    # Physical thickness of this PML (per side)
    delta = pml_width * dx
    # Roden-Gedney recommended sigma_max
    sigma_max = - (m + 1) * jnp.log(R) * c_ref / (2.0 * delta + 1e-12)

    idx = jnp.arange(n, dtype=jnp.float32)

    # distance (in cells) from the nearest boundary if in PML, else 0
    d_left  = jnp.clip(pml_width - idx, a_min=0.0)
    d_right = jnp.clip(idx - (n - 1 - pml_width), a_min=0.0)
    d = jnp.maximum(d_left, d_right)   # 0 in interior, 1..pml_width in the layer
    xi = d / jnp.maximum(pml_width, 1) # normalized [0,1]

    # polynomial grading
    sigma = sigma_max * xi**m
    kappa = 1.0 + (kappa_max - 1.0) * xi**m
    # alpha: high at inner edge, tapering to 0 at outer boundary
    alpha = alpha_max * (1.0 - xi)

    # ADE coefficients
    b = jnp.exp(- (sigma / kappa + alpha) * dt)
    # safe denominator
    denom = (sigma + kappa * alpha) * kappa + 1e-30
    a = sigma * (b - 1.0) / denom

    return {"sigma": sigma.astype(jnp.float32),
            "kappa": kappa.astype(jnp.float32),
            "alpha": alpha.astype(jnp.float32),
            "a": a.astype(jnp.float32),
            "b": b.astype(jnp.float32)}


def acoustic2D_cpml_minmem(velocity,
                           density,
                           source_i,           # (sx, sy) float indices, same convention as yours
                           f0,
                           dx, dy, dt,
                           n_steps,
                           receiver_is=None,   # (Nrec, 2) float indices
                           output_wavefield=True,
                           pml_width=10,
                           R_coeff=1e-6,
                           m=3,
                           kappa_max=3.0,
                           alpha_max=None):
    """
    2-D acoustic FD (1st-order, staggered) with true CPML via ADE.
    Memory-lean: p (nx,ny), vx (nx+1,ny), vy (nx,ny+1) + 4 memory vars.

    Positions:
      - p at cell centers (i,j) -> shape (nx, ny)
      - vx at x-faces (i+1/2, j) -> shape (nx+1, ny)
      - vy at y-faces (i, j+1/2) -> shape (nx, ny+1)

    CPML uses separate 1-D profiles on faces/centers per axis.
    """
    nx, ny = velocity.shape
    assert density.shape == velocity.shape

    # material props (float32)
    K   = (density * velocity**2).astype(jnp.float32)   # bulk modulus at centers
    rho = density.astype(jnp.float32)

    # Face densities (harmonic) for staggered updates
    # x-faces: (nx+1, ny)
    rho_L = jnp.pad(rho, ((1,0),(0,0)), mode='edge')  # left cell for face i
    rho_R = jnp.pad(rho, ((0,1),(0,0)), mode='edge')  # right cell for face i
    inv_rho_x = (2.0 / (rho_L + rho_R)).astype(jnp.float32)

    # y-faces: (nx, ny+1)
    rho_B = jnp.pad(rho, ((0,0),(1,0)), mode='edge')  # bottom cell for face j
    rho_T = jnp.pad(rho, ((0,0),(0,1)), mode='edge')  # top cell for face j
    inv_rho_y = (2.0 / (rho_B + rho_T)).astype(jnp.float32)

    # Allocate fields
    p  = jnp.zeros((nx,   ny  ), dtype=jnp.float32)
    vx = jnp.zeros((nx+1, ny  ), dtype=jnp.float32)
    vy = jnp.zeros((nx,   ny+1), dtype=jnp.float32)

    # CPML memory variables
    psi_px = jnp.zeros_like(vx)       # modifies ∂p/∂x in vx update
    psi_py = jnp.zeros_like(vy)       # modifies ∂p/∂y in vy update
    phi_vx = jnp.zeros_like(p)        # modifies ∂vx/∂x in p update
    phi_vy = jnp.zeros_like(p)        # modifies ∂vy/∂y in p update

    # Build 1-D CPML coeffs at **faces** (for dp/dx, dp/dy) and **centers** (for dv/dx, dv/dy)
    c_ref_x = velocity.max().astype(jnp.float32)
    c_ref_y = c_ref_x

    coeff_x_faces   = _cpml_1d(nx+1, dx, dt, pml_width, c_ref_x, f0, R_coeff, m, kappa_max, alpha_max)
    coeff_y_faces   = _cpml_1d(ny+1, dy, dt, pml_width, c_ref_y, f0, R_coeff, m, kappa_max, alpha_max)
    coeff_x_centers = _cpml_1d(nx,   dx, dt, pml_width, c_ref_x, f0, R_coeff, m, kappa_max, alpha_max)
    coeff_y_centers = _cpml_1d(ny,   dy, dt, pml_width, c_ref_y, f0, R_coeff, m, kappa_max, alpha_max)

    # Broadcast helpers
    ax_f = coeff_x_faces   ; ay_f = coeff_y_faces
    ax_c = coeff_x_centers ; ay_c = coeff_y_centers

    # Source (Ricker) constants
    t0     = (1.2 / f0)
    a_const= (jnp.pi * f0)**2
    # --- source bilinear weights (distinct names) ---
    sx_f, sy_f = source_i
    i0 = jnp.clip(jnp.floor(sx_f).astype(jnp.int32), 0, nx-2)
    j0 = jnp.clip(jnp.floor(sy_f).astype(jnp.int32), 0, ny-2)
    di = sx_f - i0
    dj = sy_f - j0
    ws00 = (1 - di)*(1 - dj)
    ws10 = di      *(1 - dj)
    ws01 = (1 - di)*dj
    ws11 = di      *dj

    src_amp_pa_per_s = 10e7*6

    def step(carry, it):
        p, vx, vy, psi_px, psi_py, phi_vx, phi_vy = carry
        t = it * dt

        # ===== velocities =====
        # dp/dx on x-faces -> (nx+1, ny)
        p_xpad = jnp.pad(p, ((1,1),(0,0)), mode='edge')             # (nx+2, ny)
        dpdx   = (p_xpad[1:, :] - p_xpad[:-1, :]) / dx              # (nx+1, ny)
        psi_px = ax_f["b"].reshape(-1,1) * psi_px + ax_f["a"].reshape(-1,1) * dpdx
        dpdx_tilde = dpdx / ax_f["kappa"].reshape(-1,1) + psi_px
        vx = vx - dt * inv_rho_x * dpdx_tilde

        # dp/dy on y-faces -> (nx, ny+1)
        p_ypad = jnp.pad(p, ((0,0),(1,1)), mode='edge')             # (nx, ny+2)
        dpdy   = (p_ypad[:, 1:] - p_ypad[:, :-1]) / dy              # (nx, ny+1)
        psi_py = ay_f["b"].reshape(1,-1) * psi_py + ay_f["a"].reshape(1,-1) * dpdy
        dpdy_tilde = dpdy / ay_f["kappa"].reshape(1,-1) + psi_py
        vy = vy - dt * inv_rho_y * dpdy_tilde

        # ===== pressure =====
        dvxdx = (vx[1:, :] - vx[:-1, :]) / dx                       # (nx, ny)
        phi_vx = ax_c["b"].reshape(-1,1) * phi_vx + ax_c["a"].reshape(-1,1) * dvxdx
        dvxdx_tilde = dvxdx / ax_c["kappa"].reshape(-1,1) + phi_vx

        dvydy = (vy[:, 1:] - vy[:, :-1]) / dy                       # (nx, ny)
        phi_vy = ay_c["b"].reshape(1,-1) * phi_vy + ay_c["a"].reshape(1,-1) * dvydy
        dvydy_tilde = dvydy / ay_c["kappa"].reshape(1,-1) + phi_vy

        p = p - dt * K * (dvxdx_tilde + dvydy_tilde)

        # ===== source (use ws** that were defined outside) =====
        ricker = (1.0 - 2.0 * a_const * (t - t0)**2) * jnp.exp(-a_const * (t - t0)**2)
        src = (ricker * dt).astype(jnp.float32)
        src = (dt * src_amp_pa_per_s) * ricker
        p = p.at[i0,   j0  ].add(ws00 * src)
        p = p.at[i0+1, j0  ].add(ws10 * src)
        p = p.at[i0,   j0+1].add(ws01 * src)
        p = p.at[i0+1, j0+1].add(ws11 * src)

        # ===== receivers (rename weights to avoid shadowing) =====
        out = []
        if receiver_is is not None:
            sx_rec = receiver_is[:, 0]
            sy_rec = receiver_is[:, 1]
            i0r = jnp.clip(jnp.floor(sx_rec).astype(jnp.int32), 0, nx-2)
            j0r = jnp.clip(jnp.floor(sy_rec).astype(jnp.int32), 0, ny-2)
            dir_ = sx_rec - i0r
            djr_ = sy_rec - j0r

            v00 = p[i0r,   j0r  ]
            v10 = p[i0r+1, j0r  ]
            v01 = p[i0r,   j0r+1]
            v11 = p[i0r+1, j0r+1]

            wr00 = (1-dir_)*(1-djr_)
            wr10 = dir_    *(1-djr_)
            wr01 = (1-dir_)*djr_
            wr11 = dir_    *djr_

            rec_vals = wr00*v00 + wr10*v10 + wr01*v01 + wr11*v11
            out.append(rec_vals)

        if output_wavefield:
            out.append(p)

        new_carry = (p, vx, vy, psi_px, psi_py, phi_vx, phi_vy)
        return new_carry, out

    carry0 = (p, vx, vy, psi_px, psi_py, phi_vx, phi_vy)
    _, ys = lax.scan(step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
    return ys


# ---------- helper: CPML coeffs with alpha=0 in interior ----------
def _cpml_1d_fixed(n, dx, dt, pml_width, c_ref, f0,
                   R=1e-6, m=3, kappa_max=3.0, alpha_max=None):
    if alpha_max is None:
        alpha_max = 2.0 * jnp.pi * f0
    delta = jnp.maximum(pml_width * dx, 1e-12)
    sigma_max = - (m + 1) * jnp.log(R) * c_ref / (2.0 * delta)

    idx = jnp.arange(n, dtype=jnp.float32)
    d_left  = jnp.clip(pml_width - idx, a_min=0.0)
    d_right = jnp.clip(idx - (n - 1 - pml_width), a_min=0.0)
    d  = jnp.maximum(d_left, d_right)                      # 0 in interior
    xi = d / jnp.maximum(pml_width, 1.0)

    sigma = sigma_max * xi**m
    kappa = 1.0 + (kappa_max - 1.0) * xi**m
    alpha = jnp.where(xi > 0.0, alpha_max * (1.0 - xi), 0.0)

    b = jnp.exp(- (sigma / kappa + alpha) * dt)
    denom = (sigma + kappa * alpha) * kappa + 1e-30
    a = sigma * (b - 1.0) / denom

    return {k: v.astype(jnp.float32) for k, v in
            dict(sigma=sigma, kappa=kappa, alpha=alpha, a=a, b=b).items()}

# ---------- helper: "store in half" with float32 gradients ----------
@jax.custom_vjp
def _store_f16(x):  # forward: cast to fp16; backward: pass cotangent in f32
    return lax.convert_element_type(x, jnp.float16)
def _store_f16_fwd(x):
    y = lax.convert_element_type(x, jnp.float16)
    return y, None
def _store_f16_bwd(res, g):
    return (lax.convert_element_type(g, jnp.float32),)
_store_f16.defvjp(_store_f16_fwd, _store_f16_bwd)

@jax.custom_vjp
def _store_bf16(x):
    return lax.convert_element_type(x, jnp.bfloat16)
def _store_bf16_fwd(x):
    y = lax.convert_element_type(x, jnp.bfloat16)
    return y, None
def _store_bf16_bwd(res, g):
    return (lax.convert_element_type(g, jnp.float32),)
_store_bf16.defvjp(_store_bf16_fwd, _store_bf16_bwd)

# ---------- the differentiable, strip-only CPML ----------
def acoustic2D_cpml_minmem_strips_diff(
    velocity,
    density,
    source_i,            # (sx, sy) floats in index space (cell centers)
    f0,
    dx, dy, dt,
    n_steps,
    receiver_is=None,    # (Nrec,2) floats in index space
    output_wavefield=True,
    pml_width=10,
    R_coeff=1e-6,
    m=3,
    kappa_max=3.0,
    alpha_max=None,
    src_amp_pa_per_s=3e8,
    use_bfloat16=False,
):
    nx, ny = velocity.shape
    rho = density.astype(jnp.float32)
    c   = velocity.astype(jnp.float32)
    K   = (rho * c**2).astype(jnp.float32)                 # centers

    # harmonic face inverse densities
    inv_rho_x = (2.0 / (jnp.pad(rho, ((1,0),(0,0)), 'edge') +
                        jnp.pad(rho, ((0,1),(0,0)), 'edge'))).astype(jnp.float32)  # (nx+1,ny)
    inv_rho_y = (2.0 / (jnp.pad(rho, ((0,0),(1,0)), 'edge') +
                        jnp.pad(rho, ((0,0),(0,1)), 'edge'))).astype(jnp.float32)  # (nx,ny+1)

    # fields
    p  = jnp.zeros((nx,   ny  ), jnp.float32)
    vx = jnp.zeros((nx+1, ny  ), jnp.float32)
    vy = jnp.zeros((nx,   ny+1), jnp.float32)

    # widths for faces vs centers
    pml_fx = int(min(pml_width, nx + 1))
    pml_fy = int(min(pml_width, ny + 1))
    pml_cx = int(min(pml_width, nx))
    pml_cy = int(min(pml_width, ny))

    # CPML coeffs
    c_ref = c.max().astype(jnp.float32)
    ax_f = _cpml_1d_fixed(nx+1, dx, dt, pml_width, c_ref, f0, R_coeff, m, kappa_max, alpha_max)
    ay_f = _cpml_1d_fixed(ny+1, dy, dt, pml_width, c_ref, f0, R_coeff, m, kappa_max, alpha_max)
    ax_c = _cpml_1d_fixed(nx,   dx, dt, pml_width, c_ref, f0, R_coeff, m, kappa_max, alpha_max)
    ay_c = _cpml_1d_fixed(ny,   dy, dt, pml_width, c_ref, f0, R_coeff, m, kappa_max, alpha_max)

    # choose storage op
    store_hp = _store_bf16 if use_bfloat16 else _store_f16
    hp_dtype = jnp.bfloat16 if use_bfloat16 else jnp.float16

    # memvars (stored in half; we always upcast to f32 on read)
    psi_px_L = jnp.zeros((pml_fx, ny), dtype=hp_dtype)
    psi_px_R = jnp.zeros((pml_fx, ny), dtype=hp_dtype)
    psi_py_B = jnp.zeros((nx, pml_fy), dtype=hp_dtype)
    psi_py_T = jnp.zeros((nx, pml_fy), dtype=hp_dtype)
    phi_vx_L = jnp.zeros((pml_cx, ny), dtype=hp_dtype)
    phi_vx_R = jnp.zeros((pml_cx, ny), dtype=hp_dtype)
    phi_vy_B = jnp.zeros((nx, pml_cy), dtype=hp_dtype)
    phi_vy_T = jnp.zeros((nx, pml_cy), dtype=hp_dtype)

    # precompute bilinear weights for source at centers
    sx, sy = source_i
    i0 = jnp.clip(jnp.floor(sx).astype(jnp.int32), 0, nx-2)
    j0 = jnp.clip(jnp.floor(sy).astype(jnp.int32), 0, ny-2)
    di = sx - i0
    dj = sy - j0
    ws00 = (1.0 - di) * (1.0 - dj)
    ws10 = di * (1.0 - dj)
    ws01 = (1.0 - di) * dj
    ws11 = di * dj

    # ricker
    t0 = 1.2 / f0
    a  = (jnp.pi * f0) ** 2

    # strip indices
    iL_fx = slice(0, pml_fx);           iR_fx = slice((nx+1) - pml_fx, nx+1)
    jB_fy = slice(0, pml_fy);           jT_fy = slice((ny+1) - pml_fy, ny+1)
    iL_cx = slice(0, pml_cx);           iR_cx = slice(nx - pml_cx, nx)
    jB_cy = slice(0, pml_cy);           jT_cy = slice(ny - pml_cy, ny)

    def _update_hp(prev_hp, a32, b32, deriv32):
        prev32 = lax.convert_element_type(prev_hp, jnp.float32)
        upd32  = b32 * prev32 + a32 * deriv32      # math in f32
        return store_hp(upd32), upd32               # (stored half, f32 to use now)

    def step(carry, it):
        (p, vx, vy,
         psi_px_L, psi_px_R, psi_py_B, psi_py_T,
         phi_vx_L, phi_vx_R, phi_vy_B, phi_vy_T) = carry

        t = it * dt

        # dp/dx on faces
        p_xpad = jnp.pad(p, ((1,1),(0,0)), mode='edge')
        dpdx   = (p_xpad[1:, :] - p_xpad[:-1, :]) / dx          # (nx+1, ny)

        p_ypad = jnp.pad(p, ((0,0),(1,1)), mode='edge')
        dpdy   = (p_ypad[:, 1:] - p_ypad[:, :-1]) / dy          # (nx, ny+1)

        dpdx_tilde = dpdx
        dpdy_tilde = dpdy

        if pml_fx > 0:
            aL = ax_f["a"][iL_fx][:, None]; bL = ax_f["b"][iL_fx][:, None]
            kL = ax_f["kappa"][iL_fx][:, None]
            aR = ax_f["a"][iR_fx][:, None]; bR = ax_f["b"][iR_fx][:, None]
            kR = ax_f["kappa"][iR_fx][:, None]

            psi_px_L, psi_px_L32 = _update_hp(psi_px_L, aL, bL, dpdx[iL_fx, :])
            psi_px_R, psi_px_R32 = _update_hp(psi_px_R, aR, bR, dpdx[iR_fx, :])

            dpdx_tilde = dpdx_tilde.at[iL_fx, :].set(dpdx[iL_fx, :] / kL + psi_px_L32)
            dpdx_tilde = dpdx_tilde.at[iR_fx, :].set(dpdx[iR_fx, :] / kR + psi_px_R32)

        if pml_fy > 0:
            aB = ay_f["a"][jB_fy][None, :]; bB = ay_f["b"][jB_fy][None, :]
            kB = ay_f["kappa"][jB_fy][None, :]
            aT = ay_f["a"][jT_fy][None, :]; bT = ay_f["b"][jT_fy][None, :]
            kT = ay_f["kappa"][jT_fy][None, :]

            psi_py_B, psi_py_B32 = _update_hp(psi_py_B, aB, bB, dpdy[:, jB_fy])
            psi_py_T, psi_py_T32 = _update_hp(psi_py_T, aT, bT, dpdy[:, jT_fy])

            dpdy_tilde = dpdy_tilde.at[:, jB_fy].set(dpdy[:, jB_fy] / kB + psi_py_B32)
            dpdy_tilde = dpdy_tilde.at[:, jT_fy].set(dpdy[:, jT_fy] / kT + psi_py_T32)

        # velocities
        vx = vx - dt * inv_rho_x * dpdx_tilde
        vy = vy - dt * inv_rho_y * dpdy_tilde

        # divergence at centers
        dvxdx = (vx[1:, :] - vx[:-1, :]) / dx
        dvydy = (vy[:, 1:] - vy[:, :-1]) / dy

        dvxdx_tilde = dvxdx
        dvydy_tilde = dvydy

        if pml_cx > 0:
            aL = ax_c["a"][iL_cx][:, None]; bL = ax_c["b"][iL_cx][:, None]
            kL = ax_c["kappa"][iL_cx][:, None]
            aR = ax_c["a"][iR_cx][:, None]; bR = ax_c["b"][iR_cx][:, None]
            kR = ax_c["kappa"][iR_cx][:, None]

            phi_vx_L, phi_vx_L32 = _update_hp(phi_vx_L, aL, bL, dvxdx[iL_cx, :])
            phi_vx_R, phi_vx_R32 = _update_hp(phi_vx_R, aR, bR, dvxdx[iR_cx, :])

            dvxdx_tilde = dvxdx_tilde.at[iL_cx, :].set(dvxdx[iL_cx, :] / kL + phi_vx_L32)
            dvxdx_tilde = dvxdx_tilde.at[iR_cx, :].set(dvxdx[iR_cx, :] / kR + phi_vx_R32)

        if pml_cy > 0:
            aB = ay_c["a"][jB_cy][None, :]; bB = ay_c["b"][jB_cy][None, :]
            kB = ay_c["kappa"][jB_cy][None, :]
            aT = ay_c["a"][jT_cy][None, :]; bT = ay_c["b"][jT_cy][None, :]
            kT = ay_c["kappa"][jT_cy][None, :]

            phi_vy_B, phi_vy_B32 = _update_hp(phi_vy_B, aB, bB, dvydy[:, jB_cy])
            phi_vy_T, phi_vy_T32 = _update_hp(phi_vy_T, aT, bT, dvydy[:, jT_cy])

            dvydy_tilde = dvydy_tilde.at[:, jB_cy].set(dvydy[:, jB_cy] / kB + phi_vy_B32)
            dvydy_tilde = dvydy_tilde.at[:, jT_cy].set(dvydy[:, jT_cy] / kT + phi_vy_T32)

        # pressure
        p = p - dt * K * (dvxdx_tilde + dvydy_tilde)

        # source (centered, bilinear scatter)
        ricker = (1.0 - 2.0 * a * (t - t0) ** 2) * jnp.exp(-a * (t - t0) ** 2)
        src = (dt * src_amp_pa_per_s) * ricker
        p = p.at[i0,   j0  ].add(ws00 * src)
        p = p.at[i0+1, j0  ].add(ws10 * src)
        p = p.at[i0,   j0+1].add(ws01 * src)
        p = p.at[i0+1, j0+1].add(ws11 * src)

        outs = []
        if receiver_is is not None:
            sx_rec = receiver_is[:, 0];  sy_rec = receiver_is[:, 1]
            i0r = jnp.clip(jnp.floor(sx_rec).astype(jnp.int32), 0, nx-2)
            j0r = jnp.clip(jnp.floor(sy_rec).astype(jnp.int32), 0, ny-2)
            dir_ = sx_rec - i0r;  djr_ = sy_rec - j0r
            v00 = p[i0r,   j0r  ]; v10 = p[i0r+1, j0r  ]
            v01 = p[i0r,   j0r+1]; v11 = p[i0r+1, j0r+1]
            wr00 = (1.0-dir_)*(1.0-djr_); wr10 = dir_*(1.0-djr_)
            wr01 = (1.0-dir_)*djr_;       wr11 = dir_*djr_
            rec_vals = wr00*v00 + wr10*v10 + wr01*v01 + wr11*v11
            outs.append(rec_vals)
        if output_wavefield:
            outs.append(p)

        new_carry = (p, vx, vy,
                     psi_px_L, psi_px_R, psi_py_B, psi_py_T,
                     phi_vx_L, phi_vx_R, phi_vy_B, phi_vy_T)
        return new_carry, outs

    carry0 = (p, vx, vy,
              psi_px_L, psi_px_R, psi_py_B, psi_py_T,
              phi_vx_L, phi_vx_R, phi_vy_B, phi_vy_T)

    _, ys = lax.scan(step, carry0, jnp.arange(n_steps, dtype=jnp.int32))
    return ys

def acoustic2D_pml_4th_minmem(velocity,
                             density,
                             source_i,
                             f0,
                             dx, dy, dt,
                             n_steps,
                             receiver_is=None,
                             output_wavefield=True,
                             pml_width=20,
                             R_coeff=1e-3):
    """
    4th-order accurate, memory-optimized 2D acoustic FD with boundary-only PML.

    - Uses a 5-point stencil for ∂²/∂x² and ∂²/∂y² (4th-order) to allow coarser grid.
    - Stores only two time-levels.
    - PML damping β defined only on a width-pml boundary via 1D profiles and masked application.
    """
    nx, ny = velocity.shape
    assert density.shape == (nx, ny)

    # Allocate two levels only, float32
    p_nm1 = jnp.zeros((nx, ny), jnp.float32)
    p_n   = jnp.zeros((nx, ny), jnp.float32)

    # Material properties in float32
    kappa = (density * velocity**2).astype(jnp.float32)

    # Precompute Ricker source constants
    t0      = jnp.array(1.2/f0, jnp.float32)
    factor  = jnp.array(1, jnp.float32)
    v_src   = jnp.array(velocity[source_i[0], source_i[1]], jnp.float32)
    a_const = jnp.array(jnp.pi**2 * f0**2).astype(jnp.float32)
    dt2     = jnp.array(dt*dt).astype(jnp.float32)

    # Build 1D PML damping β profiles (float32)
    sigma_max = -(3.0 * velocity.max() * jnp.log(R_coeff))
    sigma_max /= (2.0 * pml_width * dx)
    i = jnp.arange(nx, dtype=jnp.float32)
    j = jnp.arange(ny, dtype=jnp.float32)
    # quadratic ramps
    ramp_x = jnp.where(i < pml_width, (pml_width - i)/pml_width, 0.0)
    ramp_x = jnp.where(i >= nx-pml_width, (i-(nx-pml_width-1))/pml_width, ramp_x)
    ramp_y = jnp.where(j < pml_width, (pml_width - j)/pml_width, 0.0)
    ramp_y = jnp.where(j >= ny-pml_width, (j-(ny-pml_width-1))/pml_width, ramp_y)
    beta_x = (sigma_max * ramp_x**2 * dt/2.0).astype(jnp.float32)  # (nx,)
    beta_y = (sigma_max * ramp_y**2 * dt/2.0).astype(jnp.float32)  # (ny,)

    # Masks: 1 inside boundary region, 0 elsewhere
    mask_x = jnp.zeros(nx, jnp.float32)
    mask_x = mask_x.at[:pml_width].set(1.0)
    mask_x = mask_x.at[-pml_width:].set(1.0)
    mask_y = jnp.zeros(ny, jnp.float32)
    mask_y = mask_y.at[:pml_width].set(1.0)
    mask_y = mask_y.at[-pml_width:].set(1.0)

    def step(carry, it):
        p_prev, p_curr = carry
        t = it * dt

        # 4th-order in x: pad then stencil
        pad_x = jnp.pad(p_curr, ((2,2),(0,0)), mode='edge')  # shape (nx+4, ny)
        d2p_dx2 = (
            -1/12 * pad_x[:-4, :] + 4/3 * pad_x[1:-3, :]  
            -5/2 * pad_x[2:-2, :] + 4/3 * pad_x[3:-1, :]  
            -1/12 * pad_x[4:, :]
        ) / dx**2  # shape (nx, ny)

        # 4th-order in y
        pad_y = jnp.pad(p_curr, ((0,0),(2,2)), mode='edge')  # shape (nx, ny+4)
        d2p_dy2 = (
            -1/12 * pad_y[:, :-4] + 4/3 * pad_y[:, 1:-3]  
            -5/2 * pad_y[:, 2:-2] + 4/3 * pad_y[:, 3:-1]  
            -1/12 * pad_y[:, 4:]
        ) / dy**2  # shape (nx, ny)

        # Combined laplacian
        lap = dt2 * (d2p_dx2 + d2p_dy2) * kappa

        # Ricker source
        ricker = factor * (1 - 2*a_const*(t - t0)**2)
        ricker *= jnp.exp(-a_const*(t - t0)**2)
        src   = dt2 * (4*jnp.pi*(v_src**2)) * ricker

        # PML damping mask and sum
        beta_sum = beta_x[:,None] + beta_y[None,:]  # (nx,ny)
        mask2d   = mask_x[:,None] * mask_y[None,:]  # (nx,ny)
        β_mask   = beta_sum * mask2d

        # Time update with damping
        p_next = (lap + 2*p_curr - (1 - β_mask)*p_prev) / (1 + β_mask)
        p_next = p_next.at[source_i[0], source_i[1]].add(src)

        # outputs
        out = []
        if receiver_is is not None:
            out.append(p_next[receiver_is[:,0], receiver_is[:,1]])
        if output_wavefield:
            out.append(p_next)

        return (p_curr, p_next), out

    # Run the time loop
    (_, _), ys = lax.scan(step, (p_nm1, p_n), jnp.arange(n_steps))
    return ys
import jax.numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Domain
from scipy.special import hankel1
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium
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

def generate_2D_gridded_src_rec_positions(
    N=(70, 70),
    num_sources=5,
    num_receivers=5,
    margin=5.0
):
    """
    Sources ∈ [margin,   Nx-1-margin]×[margin,   Ny-1-margin]
    Receivers ∈ [dx/2, (Nx-1)-dx/2]×[dy/2, (Ny-1)-dy/2]
    """
    Nx, Ny = N

    # 1) source positions (unchanged)
    if num_sources > 1:
        src_x = jnp.linspace(margin,   Nx-1-margin,   num_sources, dtype=jnp.float32)
        src_y = jnp.linspace(margin,   Ny-1-margin,   num_sources, dtype=jnp.float32)
    else:
        cx = 0.5*((margin) + (Nx-1-margin))
        cy = 0.5*((margin) + (Ny-1-margin))
        src_x = jnp.array([cx], dtype=jnp.float32)
        src_y = jnp.array([cy], dtype=jnp.float32)

    # 2) receiver positions (full coverage minus half‐cell)
    if num_receivers > 1:
        # raw grid from 0 to Nx-1
        raw_rx = jnp.linspace(0.0, Nx-1.0, num_receivers, dtype=jnp.float32)
        raw_ry = jnp.linspace(0.0, Ny-1.0, num_receivers, dtype=jnp.float32)
        dx = raw_rx[1] - raw_rx[0]
        dy = raw_ry[1] - raw_ry[0]
        off_x = dx/2.0
        off_y = dy/2.0

        # shift by half‐spacing, then clamp in [0, Nx-1] so edges sit at ±½ cell
        recv_x = jnp.clip(raw_rx + off_x, 0.0, Nx-1.0)
        recv_y = jnp.clip(raw_ry + off_y, 0.0, Ny-1.0)
    else:
        cx = 0.5*((0.0+dx/2) + ((Nx-1.0)-dx/2))
        cy = 0.5*((0.0+dy/2) + ((Ny-1.0)-dy/2))
        recv_x = jnp.array([cx], dtype=jnp.float32)
        recv_y = jnp.array([cy], dtype=jnp.float32)

    # 3) mesh + flatten
    sx, sy = jnp.meshgrid(src_x,  src_y,  indexing='xy')
    rx, ry = jnp.meshgrid(recv_x, recv_y, indexing='xy')

    src_coords  = jnp.stack([sx.ravel(),  sy.ravel()],  axis=-1, dtype=jnp.float32)
    recv_coords = jnp.stack([rx.ravel(),  ry.ravel()], axis=-1, dtype=jnp.float32)

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
            jnp.pad((dp_dx[1:,:] - dp_dx[:-1,:]) / dx, [[1,0],[0,0]], 'constant')
        + jnp.pad((dp_dy[:,1:] - dp_dy[:,:-1]) / dy, [[0,0],[1,0]], 'constant')
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
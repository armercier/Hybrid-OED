"""2-D plane-stress elastic solver (velocity-stress, staggered grid, CPML) in JAX.

Thin-plate membrane model for the in-plane guided modes S0 and SH0.  Per unit
thickness, with plane-stress moduli

    lam_ps = 2 lam mu / (lam + 2 mu),   M = lam_ps + 2 mu,

    rho dvx/dt = dsxx/dx + dsxy/dy + fx        dsxx/dt = M dvx/dx + lam_ps dvy/dy
    rho dvy/dt = dsxy/dx + dsyy/dy + fy        dsyy/dt = lam_ps dvx/dx + M dvy/dy
                                               dsxy/dt = mu (dvx/dy + dvy/dx)

SH0 travels at vs and S0 at the plate velocity sqrt(M / rho).

Grid (index units, same convention as ``hybridoed.forward.acoustic2D_cpml_minmem``:
node i sits at x_min + i dx):
  * sxx, syy, rho, lam, mu at nodes (i, j)                 -> (nx, ny)
  * vx at (i - 1/2, j), vy at (i, j - 1/2)                 -> (nx+1, ny), (nx, ny+1)
  * sxy at (i - 1/2, j - 1/2)                              -> (nx+1, ny+1)

The source is a point force (unit direction ``force_dir``) with an arbitrary
time function sampled at the solver steps; receivers record vx and vy,
bilinearly interpolated from their staggered grids at continuous positions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from hybridoed.forward import _cpml_1d


def plane_stress_moduli(vp, vs, rho):
    mu = rho * vs**2
    lam = rho * vp**2 - 2.0 * mu
    lam_ps = 2.0 * lam * mu / (lam + 2.0 * mu)
    return lam_ps, mu


def s0_phase_velocity(f, h, vp, vs):
    """Exact Rayleigh-Lamb S0 phase velocity of a free plate of thickness h at frequency f."""
    import numpy as np
    from scipy.optimize import brentq

    w, d = 2 * np.pi * f, h / 2

    def F(c):
        k = w / c
        p, q = np.sqrt(k**2 - (w / vp) ** 2), np.sqrt((w / vs) ** 2 - k**2)
        return (q**2 - k**2) ** 2 * np.cosh(p * d) * np.sin(q * d) - 4 * k**2 * p * q * np.sinh(p * d) * np.cos(q * d)

    c_plate = 2 * vs * np.sqrt(1 - vs**2 / vp**2)
    cs = np.linspace(1.01 * vs, (1 - 1e-6) * c_plate, 2000)
    values = np.array([F(c) for c in cs])
    roots = [brentq(F, a, b) for a, b, fa, fb in zip(cs[:-1], cs[1:], values[:-1], values[1:])
             if np.sign(fa) != np.sign(fb)]
    return max(roots)


def s0_corrected_vp(vp, vs, h, f):
    """vp giving a plane-stress plate velocity equal to the exact S0 phase velocity at f (vs kept)."""
    import numpy as np

    c = s0_phase_velocity(f, h, vp, vs)
    return float(vs / np.sqrt(1 - c**2 / (4 * vs**2)))


def _bilinear_weights(pos, n0, n1):
    """Integer corner and weights for float index positions pos (..., 2)."""
    i0 = jnp.clip(jnp.floor(pos[..., 0]).astype(jnp.int32), 0, n0 - 2)
    j0 = jnp.clip(jnp.floor(pos[..., 1]).astype(jnp.int32), 0, n1 - 2)
    di, dj = pos[..., 0] - i0, pos[..., 1] - j0
    return i0, j0, ((1 - di) * (1 - dj), di * (1 - dj), (1 - di) * dj, di * dj)


def _sample(field, i0, j0, w):
    return (w[0] * field[i0, j0] + w[1] * field[i0 + 1, j0]
            + w[2] * field[i0, j0 + 1] + w[3] * field[i0 + 1, j0 + 1])


def plane_stress_cpml(vp, vs, rho, source_i, force_dir, source_time, dx, dy, dt,
                      receiver_is, f0, pml_width=10, R_coeff=1e-6, m=3, kappa_max=3.0):
    """Run the plane-stress solver; returns receiver traces (n_steps, n_rec, 2) = (vx, vy).

    vp, vs, rho: (nx, ny) node fields.  source_i, receiver_is: float node indices
    (2,) and (n_rec, 2).  source_time: (n_steps,) force time function evaluated
    at the solver times.  force_dir: (2,) force direction (fx, fy).
    """
    nx, ny = vp.shape
    lam_ps, mu = plane_stress_moduli(vp.astype(jnp.float32), vs.astype(jnp.float32),
                                     rho.astype(jnp.float32))
    M = lam_ps + 2.0 * mu
    # Staggered material: harmonic buoyancy on faces, harmonic mu on sxy points.
    pad = lambda a, w: jnp.pad(a, w, mode="edge")
    rx = pad(rho, ((1, 1), (0, 0)))
    bx = 2.0 / (rx[1:] + rx[:-1])                                    # (nx+1, ny)
    ry = pad(rho, ((0, 0), (1, 1)))
    by = 2.0 / (ry[:, 1:] + ry[:, :-1])                              # (nx, ny+1)
    mp = pad(mu, ((1, 1), (1, 1)))
    inv = 1.0 / jnp.maximum(mp, 1e-12)
    mu_xy = jnp.where(
        (mp[1:, 1:] * mp[:-1, 1:] * mp[1:, :-1] * mp[:-1, :-1]) > 0,
        4.0 / (inv[1:, 1:] + inv[:-1, 1:] + inv[1:, :-1] + inv[:-1, :-1]), 0.0)  # (nx+1, ny+1)

    c_ref = jnp.max(vp).astype(jnp.float32)
    cx_f = _cpml_1d(nx + 1, dx, dt, pml_width, c_ref, f0, R_coeff, m, kappa_max)
    cy_f = _cpml_1d(ny + 1, dy, dt, pml_width, c_ref, f0, R_coeff, m, kappa_max)
    cx_c = _cpml_1d(nx, dx, dt, pml_width, c_ref, f0, R_coeff, m, kappa_max)
    cy_c = _cpml_1d(ny, dy, dt, pml_width, c_ref, f0, R_coeff, m, kappa_max)

    def cpml(deriv, psi, coeff, axis):
        shape = (-1, 1) if axis == 0 else (1, -1)
        a, b, k = (coeff[n].reshape(shape) for n in ("a", "b", "kappa"))
        psi = b * psi + a * deriv
        return deriv / k + psi, psi

    # Source: force on vx at (sx + 1/2, sy) and on vy at (sx, sy + 1/2), scaled per cell area.
    fdir = jnp.asarray(force_dir, jnp.float32) / (dx * dy)
    si = jnp.asarray(source_i, jnp.float32)
    sx_i0, sx_j0, sx_w = _bilinear_weights(si + jnp.array([0.5, 0.0]), nx + 1, ny)
    sy_i0, sy_j0, sy_w = _bilinear_weights(si + jnp.array([0.0, 0.5]), nx, ny + 1)

    rec = jnp.asarray(receiver_is, jnp.float32)
    rx_i0, rx_j0, rx_w = _bilinear_weights(rec + jnp.array([0.5, 0.0]), nx + 1, ny)
    ry_i0, ry_j0, ry_w = _bilinear_weights(rec + jnp.array([0.0, 0.5]), nx, ny + 1)

    z = lambda s: jnp.zeros(s, jnp.float32)
    carry0 = dict(
        vx=z((nx + 1, ny)), vy=z((nx, ny + 1)), sxx=z((nx, ny)), syy=z((nx, ny)),
        sxy=z((nx + 1, ny + 1)),
        p_sxx_x=z((nx + 1, ny)), p_sxy_y=z((nx + 1, ny)),
        p_sxy_x=z((nx, ny + 1)), p_syy_y=z((nx, ny + 1)),
        p_vx_x=z((nx, ny)), p_vy_y=z((nx, ny)),
        p_vx_y=z((nx + 1, ny + 1)), p_vy_x=z((nx + 1, ny + 1)),
    )

    def step(c, f):
        c = dict(c)
        # ---- velocities ----
        d = (pad(c["sxx"], ((1, 1), (0, 0)))[1:] - pad(c["sxx"], ((1, 1), (0, 0)))[:-1]) / dx
        d_sxx_x, c["p_sxx_x"] = cpml(d, c["p_sxx_x"], cx_f, 0)                 # (nx+1, ny)
        d = (c["sxy"][:, 1:] - c["sxy"][:, :-1]) / dy
        d_sxy_y, c["p_sxy_y"] = cpml(d, c["p_sxy_y"], cy_c, 1)                 # (nx+1, ny)
        vx = c["vx"] + dt * bx * (d_sxx_x + d_sxy_y)

        d = (c["sxy"][1:, :] - c["sxy"][:-1, :]) / dx
        d_sxy_x, c["p_sxy_x"] = cpml(d, c["p_sxy_x"], cx_c, 0)                 # (nx, ny+1)
        d = (pad(c["syy"], ((0, 0), (1, 1)))[:, 1:] - pad(c["syy"], ((0, 0), (1, 1)))[:, :-1]) / dy
        d_syy_y, c["p_syy_y"] = cpml(d, c["p_syy_y"], cy_f, 1)                 # (nx, ny+1)
        vy = c["vy"] + dt * by * (d_sxy_x + d_syy_y)

        fx, fy = dt * f * fdir[0], dt * f * fdir[1]
        for k, (di, dj) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            vx = vx.at[sx_i0 + di, sx_j0 + dj].add(sx_w[k] * fx * bx[sx_i0 + di, sx_j0 + dj])
            vy = vy.at[sy_i0 + di, sy_j0 + dj].add(sy_w[k] * fy * by[sy_i0 + di, sy_j0 + dj])
        c["vx"], c["vy"] = vx, vy

        # ---- stresses ----
        d = (vx[1:] - vx[:-1]) / dx
        d_vx_x, c["p_vx_x"] = cpml(d, c["p_vx_x"], cx_c, 0)                    # (nx, ny)
        d = (vy[:, 1:] - vy[:, :-1]) / dy
        d_vy_y, c["p_vy_y"] = cpml(d, c["p_vy_y"], cy_c, 1)
        c["sxx"] = c["sxx"] + dt * (M * d_vx_x + lam_ps * d_vy_y)
        c["syy"] = c["syy"] + dt * (lam_ps * d_vx_x + M * d_vy_y)

        d = (pad(vx, ((0, 0), (1, 1)))[:, 1:] - pad(vx, ((0, 0), (1, 1)))[:, :-1]) / dy
        d_vx_y, c["p_vx_y"] = cpml(d, c["p_vx_y"], cy_f, 1)                    # (nx+1, ny+1)
        d = (pad(vy, ((1, 1), (0, 0)))[1:] - pad(vy, ((1, 1), (0, 0)))[:-1]) / dx
        d_vy_x, c["p_vy_x"] = cpml(d, c["p_vy_x"], cx_f, 0)
        c["sxy"] = c["sxy"] + dt * mu_xy * (d_vx_y + d_vy_x)

        out = jnp.stack([_sample(vx, rx_i0, rx_j0, rx_w), _sample(vy, ry_i0, ry_j0, ry_w)], -1)
        return c, out

    _, traces = lax.scan(step, carry0, jnp.asarray(source_time, jnp.float32))
    return traces


def plane_stress_free(vp, vs, rho, source_i, force_dir, source_time, dx, dy, dt, receiver_is,
                      checkpoint_every=0):
    """Plane-stress solver for a finite plate with traction-free edges (no absorption).

    The plate occupies the nodes i = 0..nx-1, j = 0..ny-1 (edges on the node lines
    x = 0, x = (nx-1) dx, ...).  Stress imaging: normal stress is zero on edge nodes,
    sxy is antisymmetric across the edges, and the in-edge normal stress of edge
    nodes follows from the zero-normal-stress constraint.  Same staggering, source
    and receiver conventions as ``plane_stress_cpml``; returns (n_steps, n_rec, 2).

    ``checkpoint_every > 0`` rematerialises the time loop in chunks of that many
    steps (only chunk-boundary states are stored for reverse-mode gradients).
    """
    nx, ny = vp.shape
    lam, mu = plane_stress_moduli(vp.astype(jnp.float32), vs.astype(jnp.float32),
                                  rho.astype(jnp.float32))
    M = lam + 2.0 * mu
    M_edge = M - lam**2 / M  # in-edge modulus when the edge-normal stress vanishes
    pad = lambda a, w: jnp.pad(a, w, mode="edge")
    rx = pad(rho, ((1, 1), (0, 0)))
    bx = 2.0 / (rx[1:] + rx[:-1])                                    # (nx+1, ny)
    ry = pad(rho, ((0, 0), (1, 1)))
    by = 2.0 / (ry[:, 1:] + ry[:, :-1])                              # (nx, ny+1)
    inv = 1.0 / mu
    mu_xy = jnp.zeros((nx + 1, ny + 1), jnp.float32).at[1:-1, 1:-1].set(
        4.0 / (inv[1:, 1:] + inv[:-1, 1:] + inv[1:, :-1] + inv[:-1, :-1]))

    i_edge = jnp.zeros((nx, ny), bool).at[0].set(True).at[-1].set(True)
    j_edge = jnp.zeros((nx, ny), bool).at[:, 0].set(True).at[:, -1].set(True)

    fdir = jnp.asarray(force_dir, jnp.float32) / (dx * dy)
    si = jnp.asarray(source_i, jnp.float32)
    sx_i0, sx_j0, sx_w = _bilinear_weights(si + jnp.array([0.5, 0.0]), nx + 1, ny)
    sy_i0, sy_j0, sy_w = _bilinear_weights(si + jnp.array([0.0, 0.5]), nx, ny + 1)
    rec = jnp.asarray(receiver_is, jnp.float32)
    rx_i0, rx_j0, rx_w = _bilinear_weights(rec + jnp.array([0.5, 0.0]), nx + 1, ny)
    ry_i0, ry_j0, ry_w = _bilinear_weights(rec + jnp.array([0.0, 0.5]), nx, ny + 1)

    z = lambda s: jnp.zeros(s, jnp.float32)
    carry0 = (z((nx + 1, ny)), z((nx, ny + 1)), z((nx, ny)), z((nx, ny)), z((nx + 1, ny + 1)))

    def step(c, f):
        vx, vy, sxx, syy, sxy = c
        # shear stress antisymmetric across the edges (outside rows/columns are images)
        sxy = sxy.at[0].set(-sxy[1]).at[-1].set(-sxy[-2])
        sxy = sxy.at[:, 0].set(-sxy[:, 1]).at[:, -1].set(-sxy[:, -2])
        # velocities on interior faces (outermost faces lie outside the plate and stay 0)
        vx = vx.at[1:-1].add(dt * bx[1:-1] * ((sxx[1:] - sxx[:-1]) / dx
                                             + (sxy[1:-1, 1:] - sxy[1:-1, :-1]) / dy))
        vy = vy.at[:, 1:-1].add(dt * by[:, 1:-1] * ((sxy[1:, 1:-1] - sxy[:-1, 1:-1]) / dx
                                                   + (syy[:, 1:] - syy[:, :-1]) / dy))
        fx, fy = dt * f * fdir[0], dt * f * fdir[1]
        for k, (di, dj) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            vx = vx.at[sx_i0 + di, sx_j0 + dj].add(sx_w[k] * fx * bx[sx_i0 + di, sx_j0 + dj])
            vy = vy.at[sy_i0 + di, sy_j0 + dj].add(sy_w[k] * fy * by[sy_i0 + di, sy_j0 + dj])
        # normal stresses
        exx = (vx[1:] - vx[:-1]) / dx
        eyy = (vy[:, 1:] - vy[:, :-1]) / dy
        sxx_new = sxx + dt * (M * exx + lam * eyy)
        syy_new = syy + dt * (lam * exx + M * eyy)
        sxx_new = jnp.where(i_edge, 0.0, jnp.where(j_edge, sxx + dt * M_edge * exx, sxx_new))
        syy_new = jnp.where(j_edge, 0.0, jnp.where(i_edge, syy + dt * M_edge * eyy, syy_new))
        # shear stress at interior (i-1/2, j-1/2) points
        sxy = sxy.at[1:-1, 1:-1].add(dt * mu_xy[1:-1, 1:-1] * (
            (vx[1:-1, 1:] - vx[1:-1, :-1]) / dy + (vy[1:, 1:-1] - vy[:-1, 1:-1]) / dx))
        out = jnp.stack([_sample(vx, rx_i0, rx_j0, rx_w), _sample(vy, ry_i0, ry_j0, ry_w)], -1)
        return (vx, vy, sxx_new, syy_new, sxy), out

    return _scan_maybe_checkpointed(step, carry0, source_time, checkpoint_every)


def _scan_maybe_checkpointed(step, carry0, source_time, checkpoint_every):
    src = jnp.asarray(source_time, jnp.float32)
    if not checkpoint_every:
        return lax.scan(step, carry0, src)[1]
    n = src.shape[0]
    n_chunks = -(-n // checkpoint_every)
    chunks = jnp.pad(src, (0, n_chunks * checkpoint_every - n)).reshape(n_chunks, checkpoint_every)

    @jax.checkpoint
    def chunk(carry, s):
        return lax.scan(step, carry, s)

    _, traces = lax.scan(chunk, carry0, chunks)
    return traces.reshape(n_chunks * checkpoint_every, *traces.shape[2:])[:n]

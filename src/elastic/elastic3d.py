"""3-D isotropic elastic solver for a free plate (velocity-stress, staggered grid) in JAX.

The plate occupies the nodes i = 0..nx-1, j = 0..ny-1, k = 0..nz-1 (faces on the
outer node planes) and is traction-free on all six faces (stress imaging):

  * normal stress normal to a face vanishes on that face's node plane; at edge and
    corner nodes all constrained normal stresses vanish together, and the remaining
    normal stresses use the strains consistent with those constraints;
  * shear stresses are antisymmetric across the faces (image rows outside the plate).

Grid (index units): sxx, syy, szz, material at nodes (i, j, k); vx at (i-1/2, j, k),
vy at (i, j-1/2, k), vz at (i, j, k-1/2); sxy at (i-1/2, j-1/2, k), sxz at
(i-1/2, j, k-1/2), syz at (i, j-1/2, k-1/2).  Material is given as in-plane maps
(nx, ny), uniform through the thickness.  The source is a point force with
trilinear injection; receivers record (vx, vy, vz) on the top face k = nz-1.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def _harmonic_pairs(mu, axes):
    """Harmonic mean of mu over the 4 nodes around each half-point in the two given axes."""
    a0, a1 = axes
    s = lambda a, ax, lo: lax.slice_in_dim(a, 1 if lo else 0, a.shape[ax] - (0 if lo else 1), axis=ax)
    inv = 1.0 / mu
    tot = 0.0
    for lo0 in (True, False):
        for lo1 in (True, False):
            tot = tot + s(s(inv, a0, lo0), a1, lo1)
    return 4.0 / tot


def _trilinear(pos, shape):
    idx, w = [], []
    for d in range(3):
        i0 = jnp.clip(jnp.floor(pos[d]).astype(jnp.int32), 0, shape[d] - 2)
        idx.append(i0)
        w.append(pos[d] - i0)
    out = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                weight = ((w[0] if a else 1 - w[0]) * (w[1] if b else 1 - w[1]) * (w[2] if c else 1 - w[2]))
                out.append(((idx[0] + a, idx[1] + b, idx[2] + c), weight))
    return out


def _bilinear_plane(field, pos, k):
    """Sample field[:, :, k] at float in-plane positions pos (n, 2)."""
    i0 = jnp.clip(jnp.floor(pos[:, 0]).astype(jnp.int32), 0, field.shape[0] - 2)
    j0 = jnp.clip(jnp.floor(pos[:, 1]).astype(jnp.int32), 0, field.shape[1] - 2)
    di, dj = pos[:, 0] - i0, pos[:, 1] - j0
    f = field[:, :, k]
    return ((1 - di) * (1 - dj) * f[i0, j0] + di * (1 - dj) * f[i0 + 1, j0]
            + (1 - di) * dj * f[i0, j0 + 1] + di * dj * f[i0 + 1, j0 + 1])


def plate3d_free(vp, vs, rho, nz, source_i, force, source_time, dx, dz, dt, receiver_is):
    """Run the 3-D free-plate solver; returns top-face traces (n_steps, n_rec, 3)."""
    nx, ny = vp.shape
    ext = lambda a: jnp.broadcast_to(jnp.asarray(a, jnp.float32)[:, :, None], (nx, ny, nz))
    rho3, mu3 = ext(rho), ext(rho * vs**2)
    lam3 = ext(rho * vp**2) - 2.0 * mu3
    b_x = 2.0 / (rho3[1:] + rho3[:-1])                               # interior x-faces
    b_y = 2.0 / (rho3[:, 1:] + rho3[:, :-1])
    b_z = 2.0 / (rho3[:, :, 1:] + rho3[:, :, :-1])
    mu_xy = _harmonic_pairs(mu3, (0, 1))                              # (nx-1, ny-1, nz)
    mu_xz = _harmonic_pairs(mu3, (0, 2))                              # (nx-1, ny, nz-1)
    mu_yz = _harmonic_pairs(mu3, (1, 2))                              # (nx, ny-1, nz-1)

    edge = lambda n: jnp.zeros(n, bool).at[0].set(True).at[-1].set(True)
    bx = edge(nx)[:, None, None] & jnp.ones((nx, ny, nz), bool)
    by = edge(ny)[None, :, None] & jnp.ones((nx, ny, nz), bool)
    bz = edge(nz)[None, None, :] & jnp.ones((nx, ny, nz), bool)
    nb = bx.astype(jnp.float32) + by + bz
    tr_factor = 2.0 * mu3 / (lam3 * nb + 2.0 * mu3)

    si = jnp.asarray(source_i, jnp.float32)
    f = jnp.asarray(force, jnp.float32) / (dx * dx * dz)
    inj = [
        _trilinear(si + jnp.array([0.5, 0.0, 0.0]), (nx + 1, ny, nz)),
        _trilinear(si + jnp.array([0.0, 0.5, 0.0]), (nx, ny + 1, nz)),
        _trilinear(si + jnp.array([0.0, 0.0, 0.5]), (nx, ny, nz + 1)),
    ]
    rec = jnp.asarray(receiver_is, jnp.float32)
    rho_src = rho3[tuple(jnp.clip(jnp.round(si).astype(jnp.int32), 0, jnp.array([nx, ny, nz]) - 1))]

    z = lambda s: jnp.zeros(s, jnp.float32)
    carry0 = (z((nx + 1, ny, nz)), z((nx, ny + 1, nz)), z((nx, ny, nz + 1)),
              z((nx, ny, nz)), z((nx, ny, nz)), z((nx, ny, nz)),
              z((nx + 1, ny + 1, nz)), z((nx + 1, ny, nz + 1)), z((nx, ny + 1, nz + 1)))

    def step(c, s):
        vx, vy, vz, sxx, syy, szz, sxy, sxz, syz = c
        # antisymmetric shear images outside the plate
        sxy = sxy.at[0].set(-sxy[1]).at[-1].set(-sxy[-2]).at[:, 0].set(-sxy[:, 1]).at[:, -1].set(-sxy[:, -2])
        sxz = sxz.at[0].set(-sxz[1]).at[-1].set(-sxz[-2]).at[:, :, 0].set(-sxz[:, :, 1]).at[:, :, -1].set(-sxz[:, :, -2])
        syz = syz.at[:, 0].set(-syz[:, 1]).at[:, -1].set(-syz[:, -2]).at[:, :, 0].set(-syz[:, :, 1]).at[:, :, -1].set(-syz[:, :, -2])
        # velocities on interior faces
        vx = vx.at[1:-1].add(dt * b_x * ((sxx[1:] - sxx[:-1]) / dx + (sxy[1:-1, 1:] - sxy[1:-1, :-1]) / dx
                                         + (sxz[1:-1, :, 1:] - sxz[1:-1, :, :-1]) / dz))
        vy = vy.at[:, 1:-1].add(dt * b_y * ((sxy[1:, 1:-1] - sxy[:-1, 1:-1]) / dx + (syy[:, 1:] - syy[:, :-1]) / dx
                                            + (syz[:, 1:-1, 1:] - syz[:, 1:-1, :-1]) / dz))
        vz = vz.at[:, :, 1:-1].add(dt * b_z * ((sxz[1:, :, 1:-1] - sxz[:-1, :, 1:-1]) / dx
                                               + (syz[:, 1:, 1:-1] - syz[:, :-1, 1:-1]) / dx
                                               + (szz[:, :, 1:] - szz[:, :, :-1]) / dz))
        v = [vx, vy, vz]
        for comp in range(3):
            for (ijk, w) in inj[comp]:
                v[comp] = v[comp].at[ijk].add(dt * s * f[comp] * w / rho_src)
        vx, vy, vz = v
        # normal stresses with the free-face constraints
        exx = (vx[1:] - vx[:-1]) / dx
        eyy = (vy[:, 1:] - vy[:, :-1]) / dx
        ezz = (vz[:, :, 1:] - vz[:, :, :-1]) / dz
        s_free = jnp.where(bx, 0.0, exx) + jnp.where(by, 0.0, eyy) + jnp.where(bz, 0.0, ezz)
        tr = s_free * tr_factor
        sxx = jnp.where(bx, 0.0, sxx + dt * (lam3 * tr + 2.0 * mu3 * exx))
        syy = jnp.where(by, 0.0, syy + dt * (lam3 * tr + 2.0 * mu3 * eyy))
        szz = jnp.where(bz, 0.0, szz + dt * (lam3 * tr + 2.0 * mu3 * ezz))
        # shear stresses at interior half-points
        sxy = sxy.at[1:-1, 1:-1].add(dt * mu_xy * ((vx[1:-1, 1:] - vx[1:-1, :-1]) / dx + (vy[1:, 1:-1] - vy[:-1, 1:-1]) / dx))
        sxz = sxz.at[1:-1, :, 1:-1].add(dt * mu_xz * ((vx[1:-1, :, 1:] - vx[1:-1, :, :-1]) / dz
                                                      + (vz[1:, :, 1:-1] - vz[:-1, :, 1:-1]) / dx))
        syz = syz.at[:, 1:-1, 1:-1].add(dt * mu_yz * ((vy[:, 1:-1, 1:] - vy[:, 1:-1, :-1]) / dz
                                                      + (vz[:, 1:, 1:-1] - vz[:, :-1, 1:-1]) / dx))
        top = nz - 1
        out = jnp.stack([
            _bilinear_plane(vx, rec + jnp.array([0.5, 0.0]), top),
            _bilinear_plane(vy, rec + jnp.array([0.0, 0.5]), top),
            1.5 * _bilinear_plane(vz, rec, top) - 0.5 * _bilinear_plane(vz, rec, top - 1),  # face k-1/2 -> surface
        ], -1)
        return (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz), out

    _, traces = lax.scan(step, carry0, jnp.asarray(source_time, jnp.float32))
    return traces

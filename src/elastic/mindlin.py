"""2-D Mindlin (first-order shear deformation) plate solver in JAX, for flexural (A0) waves.

Velocity / stress-resultant form, per unit area of the mid-surface:

    rho h   d(vz)/dt = dQx/dx + dQy/dy + q
    rho I   d(Ox)/dt = dMxx/dx + dMxy/dy - Qx        I = h^3 / 12
    rho I   d(Oy)/dt = dMxy/dx + dMyy/dy - Qy
    d(Mxx)/dt = D (dOx/dx + nu dOy/dy),  d(Myy)/dt = D (nu dOx/dx + dOy/dy)
    d(Mxy)/dt = D (1 - nu) / 2 (dOx/dy + dOy/dx)
    d(Qx)/dt  = kappa G h (dvz/dx + Ox),  d(Qy)/dt = kappa G h (dvz/dy + Oy)

with D = E h^3 / (12 (1 - nu^2)), kappa = pi^2 / 12 (Mindlin), vz the deflection
velocity and Ox, Oy the rotation rates. Surface in-plane velocity = +/- (h/2) O.

Grid (index units, as in plane_stress.py): vz, Mxx, Myy, material at nodes (i, j);
Ox, Qx at (i-1/2, j); Oy, Qy at (i, j-1/2); Mxy at (i-1/2, j-1/2).
Edges are free (Mnn = Mns = Qn = 0, by imaging as in ``plane_stress_free``); an
optional sponge of width ``sponge_cells`` damps all fields near the edges to mimic
an open plate. Source: normal point force; receivers: vz (and optionally the
top-surface in-plane velocity) by bilinear interpolation.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

from plane_stress import _bilinear_weights, _sample, _scan_maybe_checkpointed

KAPPA = np.pi**2 / 12.0  # Mindlin's original shear correction


def plate_moduli(vp, vs, rho, h):
    mu = rho * vs**2
    lam = rho * vp**2 - 2.0 * mu
    nu = lam / (2.0 * (lam + mu))
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    return D, nu, mu


def exact_a0_phase_velocity(f, h, vp, vs):
    """Exact Rayleigh-Lamb A0 phase velocity of a free plate (thickness h) at frequency f."""
    from scipy.optimize import brentq

    w, d = 2 * np.pi * f, h / 2

    def F(c):
        k = w / c
        p = np.sqrt(k**2 - (w / vp) ** 2)
        q2 = (w / vs) ** 2 - k**2
        if q2 > 0:
            q = np.sqrt(q2)
            return (q**2 - k**2) ** 2 * np.sinh(p * d) * np.cos(q * d) + 4 * k**2 * p * q * np.cosh(p * d) * np.sin(q * d)
        qq = np.sqrt(-q2)
        return (qq**2 + k**2) ** 2 * np.sinh(p * d) * np.cosh(qq * d) - 4 * k**2 * p * qq * np.cosh(p * d) * np.sinh(qq * d)

    cs = np.linspace(30.0, 0.999 * vs, 4000)
    v = np.array([F(c) for c in cs])
    return min(brentq(F, a, b) for a, b, fa, fb in zip(cs[:-1], cs[1:], v[:-1], v[1:]) if np.sign(fa) != np.sign(fb))


def mindlin_phase_velocity(f, h, vp, vs, rho, kappa):
    """Flexural-branch phase velocity of the Mindlin plate."""
    D, nu, mu = (float(x) for x in plate_moduli(vp, vs, rho, h))
    w, kG, rI = 2 * np.pi * f, kappa * mu * h, rho * h**3 / 12
    a, b = kG * D, kG * (kG - rI * w**2) - rho * h * w**2 * D - kG**2
    c = -rho * h * w**2 * (kG - rI * w**2)
    return w / np.sqrt((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a))


def a0_matched_kappa(h, vp, vs, rho, f):
    """Shear correction factor making the Mindlin A0 phase velocity exact at frequency f."""
    from scipy.optimize import brentq

    target = exact_a0_phase_velocity(f, h, vp, vs)
    return brentq(lambda k: mindlin_phase_velocity(f, h, vp, vs, rho, k) - target, 0.5, 1.2)


def mindlin_plate(vp, vs, rho, h, source_i, source_time, dx, dy, dt, receiver_is,
                  sponge_cells=0, sponge_strength=0.05, surface_inplane=False, checkpoint_every=0,
                  kappa=KAPPA, mass_scale=None, inertia_scale=None, stiffness_scale=None, damping=0.0,
                  rigid_mask=None, rigid_added_mass=0.0, rigid_added_inertia=0.0):
    """Returns traces (n_steps, n_rec, 1 or 3): vz [, top-surface vx, vy].

    ``kappa``: shear correction (pi^2/12 by default; ``a0_matched_kappa`` makes A0
    dispersion exact at a chosen frequency, within 0.1 % over 12.5-50 kHz for 10 mm PMMA).
    ``mass_scale`` / ``inertia_scale`` / ``stiffness_scale``: optional (nx, ny) factors on
    rho h, rho I and (D, kappa G h), e.g. an equivalent patch for a bonded mass.
    ``damping``: rate a (1/s); the velocities are multiplied by (1 - a dt) each step
    (mass-proportional damping: amplitude decays as exp(-a t / 2), Q ~ omega / a).
    ``rigid_mask``: optional (nx, ny) bool footprint of a rigid body bonded to the plate
    (e.g. a steel block). Its motion is restricted to deflection + two rotations about the
    footprint centre: after each velocity update the impulse on the footprint is projected
    onto these 3 rigid modes with the footprint's plate mass / rotary inertia plus
    ``rigid_added_mass`` (kg) and ``rigid_added_inertia`` (kg m^2, about the mid-plane,
    same for both axes), and the footprint velocities are reset to that rigid motion.
    In-plane translation and twist of the body are not represented.
    """
    nx, ny = vp.shape
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    vp, vs, rho, h = f32(vp), f32(vs), f32(rho), f32(jnp.broadcast_to(h, (nx, ny)))
    D, nu, mu = plate_moduli(vp, vs, rho, h)
    rhoh, rhoI, kGh = rho * h, rho * h**3 / 12.0, kappa * mu * h
    if mass_scale is not None:
        rhoh = rhoh * f32(mass_scale)
    if inertia_scale is not None:
        rhoI = rhoI * f32(inertia_scale)
    if stiffness_scale is not None:
        D, kGh = D * f32(stiffness_scale), kGh * f32(stiffness_scale)
    D_edge = D * (1.0 - nu**2)
    decay = 1.0 - damping * dt

    rigid = rigid_mask is not None
    if rigid:
        # traceable: the mask may be a JAX array (e.g. a searched position)
        mn_j = jnp.asarray(rigid_mask, bool)
        mfx_j = jnp.zeros((nx + 1, ny), bool).at[1:-1].set(mn_j[1:] & mn_j[:-1])
        mfy_j = jnp.zeros((nx, ny + 1), bool).at[:, 1:-1].set(mn_j[:, 1:] & mn_j[:, :-1])
        wsum = jnp.sum(mn_j)
        ic = jnp.sum(jnp.arange(nx)[:, None] * mn_j) / wsum
        jc = jnp.sum(jnp.arange(ny)[None, :] * mn_j) / wsum
        xn = (jnp.arange(nx, dtype=jnp.float32)[:, None] - ic) * dx * jnp.ones((1, ny), jnp.float32)
        yn = (jnp.arange(ny, dtype=jnp.float32)[None, :] - jc) * dy * jnp.ones((nx, 1), jnp.float32)
        m_node = jnp.where(mn_j, rhoh * dx * dy, 0.0)
        ep = lambda a, w: jnp.pad(a, w, mode="edge")
        j_fx = jnp.where(mfx_j, (0.5 * (ep(rhoI, ((1, 1), (0, 0)))[1:] + ep(rhoI, ((1, 1), (0, 0)))[:-1])) * dx * dy, 0.0)
        j_fy = jnp.where(mfy_j, (0.5 * (ep(rhoI, ((0, 0), (1, 1)))[:, 1:] + ep(rhoI, ((0, 0), (1, 1)))[:, :-1])) * dx * dy, 0.0)
        M_tot = jnp.sum(m_node) + rigid_added_mass
        I_a = jnp.sum(m_node * xn**2) + jnp.sum(j_fx) + rigid_added_inertia
        I_b = jnp.sum(m_node * yn**2) + jnp.sum(j_fy) + rigid_added_inertia

    pad = lambda a, w: jnp.pad(a, w, mode="edge")
    avg_x = lambda a: 0.5 * (pad(a, ((1, 1), (0, 0)))[1:] + pad(a, ((1, 1), (0, 0)))[:-1])        # (nx+1, ny)
    avg_y = lambda a: 0.5 * (pad(a, ((0, 0), (1, 1)))[:, 1:] + pad(a, ((0, 0), (1, 1)))[:, :-1])  # (nx, ny+1)
    harm_x = lambda a: 1.0 / avg_x(1.0 / a)
    harm_y = lambda a: 1.0 / avg_y(1.0 / a)
    rhoI_x, rhoI_y = avg_x(rhoI), avg_y(rhoI)
    kGh_x, kGh_y = harm_x(kGh), harm_y(kGh)
    Dt = D * (1.0 - nu) / 2.0
    inv = 1.0 / Dt
    D_xy = jnp.zeros((nx + 1, ny + 1), jnp.float32).at[1:-1, 1:-1].set(
        4.0 / (inv[1:, 1:] + inv[:-1, 1:] + inv[1:, :-1] + inv[:-1, :-1]))

    i_edge = jnp.zeros((nx, ny), bool).at[0].set(True).at[-1].set(True)
    j_edge = jnp.zeros((nx, ny), bool).at[:, 0].set(True).at[:, -1].set(True)

    if sponge_cells:
        def prof(n, offset):
            pos = np.arange(n) + offset
            d = np.maximum(np.maximum(sponge_cells - pos, pos - (n - 1 + 2 * offset - sponge_cells)), 0)
            return np.exp(-sponge_strength * (np.minimum(d, sponge_cells) / sponge_cells) ** 2).astype(np.float32)
        gxn, gyn = prof(nx, 0.0), prof(ny, 0.0)
        gxf, gyf = prof(nx + 1, -0.5), prof(ny + 1, -0.5)
        g_node = jnp.asarray(gxn[:, None] * gyn[None, :])
        g_fx = jnp.asarray(gxf[:, None] * gyn[None, :])
        g_fy = jnp.asarray(gxn[:, None] * gyf[None, :])
        g_c = jnp.asarray(gxf[:, None] * gyf[None, :])
    else:
        g_node = g_fx = g_fy = g_c = None

    si = jnp.asarray(source_i, jnp.float32)
    s_i0, s_j0, s_w = _bilinear_weights(si, nx, ny)
    src_scale = dt / (rhoh[s_i0, s_j0] * dx * dy)
    rec = jnp.asarray(receiver_is, jnp.float32)
    r_i0, r_j0, r_w = _bilinear_weights(rec, nx, ny)
    if surface_inplane:
        rx = _bilinear_weights(rec + jnp.array([0.5, 0.0]), nx + 1, ny)
        ry = _bilinear_weights(rec + jnp.array([0.0, 0.5]), nx, ny + 1)
        h_rec = _sample(h, r_i0, r_j0, r_w)

    z = lambda s: jnp.zeros(s, jnp.float32)
    carry0 = (z((nx, ny)), z((nx + 1, ny)), z((nx, ny + 1)), z((nx, ny)), z((nx, ny)),
              z((nx + 1, ny + 1)), z((nx + 1, ny)), z((nx, ny + 1)), z((3,)))

    def step(c, f):
        vz, Ox, Oy, Mxx, Myy, Mxy, Qx, Qy, body = c
        vz0, Ox0, Oy0 = vz, Ox, Oy
        # free-edge images: twisting moment and normal shear forces antisymmetric across edges
        Mxy = Mxy.at[0].set(-Mxy[1]).at[-1].set(-Mxy[-2]).at[:, 0].set(-Mxy[:, 1]).at[:, -1].set(-Mxy[:, -2])
        Qx = Qx.at[0].set(-Qx[1]).at[-1].set(-Qx[-2])
        Qy = Qy.at[:, 0].set(-Qy[:, 1]).at[:, -1].set(-Qy[:, -2])
        # velocities
        vz = vz + dt / rhoh * ((Qx[1:] - Qx[:-1]) / dx + (Qy[:, 1:] - Qy[:, :-1]) / dy)
        for k, (di, dj) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            vz = vz.at[s_i0 + di, s_j0 + dj].add(s_w[k] * f * src_scale)
        Ox = Ox.at[1:-1].add(dt / rhoI_x[1:-1] * ((Mxx[1:] - Mxx[:-1]) / dx
                                                  + (Mxy[1:-1, 1:] - Mxy[1:-1, :-1]) / dy - Qx[1:-1]))
        Oy = Oy.at[:, 1:-1].add(dt / rhoI_y[:, 1:-1] * ((Mxy[1:, 1:-1] - Mxy[:-1, 1:-1]) / dx
                                                        + (Myy[:, 1:] - Myy[:, :-1]) / dy - Qy[:, 1:-1]))
        if rigid:  # project the footprint impulse onto the rigid modes (V, Wa, Wb)
            dvz, dOx, dOy = vz - vz0, Ox - Ox0, Oy - Oy0
            F = jnp.sum(m_node * dvz)
            Ma = jnp.sum(m_node * dvz * xn) - jnp.sum(j_fx * dOx)
            Mb = jnp.sum(m_node * dvz * yn) - jnp.sum(j_fy * dOy)
            body = body + jnp.stack([F / M_tot, Ma / I_a, Mb / I_b])
            vz = jnp.where(mn_j, body[0] + body[1] * xn + body[2] * yn, vz)
            Ox = jnp.where(mfx_j, -body[1], Ox)
            Oy = jnp.where(mfy_j, -body[2], Oy)
        vz, Ox, Oy, body = vz * decay, Ox * decay, Oy * decay, body * decay
        # moments (normal moment vanishes on its edge)
        kx = (Ox[1:] - Ox[:-1]) / dx
        ky = (Oy[:, 1:] - Oy[:, :-1]) / dy
        Mxx_new = jnp.where(i_edge, 0.0, jnp.where(j_edge, Mxx + dt * D_edge * kx, Mxx + dt * D * (kx + nu * ky)))
        Myy_new = jnp.where(j_edge, 0.0, jnp.where(i_edge, Myy + dt * D_edge * ky, Myy + dt * D * (nu * kx + ky)))
        Mxy = Mxy.at[1:-1, 1:-1].add(dt * D_xy[1:-1, 1:-1] * (
            (Ox[1:-1, 1:] - Ox[1:-1, :-1]) / dy + (Oy[1:, 1:-1] - Oy[:-1, 1:-1]) / dx))
        # shear forces
        Qx = Qx.at[1:-1].add(dt * kGh_x[1:-1] * ((vz[1:] - vz[:-1]) / dx + Ox[1:-1]))
        Qy = Qy.at[:, 1:-1].add(dt * kGh_y[:, 1:-1] * ((vz[:, 1:] - vz[:, :-1]) / dy + Oy[:, 1:-1]))
        if g_node is not None:
            vz, Mxx_new, Myy_new = vz * g_node, Mxx_new * g_node, Myy_new * g_node
            Ox, Qx, Oy, Qy, Mxy = Ox * g_fx, Qx * g_fx, Oy * g_fy, Qy * g_fy, Mxy * g_c
        outs = [_sample(vz, r_i0, r_j0, r_w)]
        if surface_inplane:  # top surface (z = +h/2): u = z * phi
            outs += [0.5 * h_rec * _sample(Ox, *rx), 0.5 * h_rec * _sample(Oy, *ry)]
        return (vz, Ox, Oy, Mxx_new, Myy_new, Mxy, Qx, Qy, body), jnp.stack(outs, -1)

    return _scan_maybe_checkpointed(step, carry0, source_time, checkpoint_every)

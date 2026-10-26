"""Forward comparison of the plane-stress JAX solver with the Salvus 3-D plate data.

Inputs (read-only):
  * src/scripts/INTERNAL/ENTITIES/SIMULATION_CONFIGURATIONS/test_25kHz.json : wavelet
  * src/scripts/INTERNAL/MESHES/.../mesh.h5 : RHO, VP, VS at the GLL points
  * src/scripts/extracted_velocity_wavefields_S0/event_*.nc : Salvus in-plane velocity

The plane-stress model uses the thickness average of the plane-stress moduli,
weighted with the GLL quadrature weights of the single through-thickness element
(i.e. the membrane stiffness the Salvus discretisation integrates).

Usage:  python src/elastic/compare_salvus.py --dx 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "src" / "elastic"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SCRIPTS = REPO / "src" / "scripts"
WAVELET_JSON = SCRIPTS / "INTERNAL/ENTITIES/SIMULATION_CONFIGURATIONS/test_25kHz.json"
SALVUS_DIR = SCRIPTS / "extracted_velocity_wavefields_S0"
RESULTS = REPO / "elastic_results"

L = 1.1                      # plate side (m), centred on 0
ABC_WIDTH = 0.066            # Salvus absorbing-layer width (mesh attribute max_dist_ABC)
INTERIOR = 0.45              # compare only inside the non-absorbing region
GLL5_WEIGHTS = np.array([0.1, 49 / 90, 32 / 45, 49 / 90, 0.1]) / 2.0  # on [0, 1]


# ----------------------------------------------------------------------------- inputs


def load_wavelet():
    cfg = json.loads(WAVELET_JSON.read_text())
    w = cfg["arguments"]["event_configuration"]["arguments"]["wavelet"]
    assert w["class_name"].endswith("stf.Ricker"), w["class_name"]
    f0 = float(w["arguments"]["center_frequency"])
    shift = w["arguments"]["time_shift_in_seconds"]
    return f0, (0.0 if shift is None else float(shift))


def ricker(t, f0, shift):
    a = (np.pi * f0 * (t - shift)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def mesh_path() -> Path:
    cfg = json.loads(WAVELET_JSON.read_text())
    return SCRIPTS / cfg["arguments"]["hdf5_filename"]


def effective_plane_stress_model(x_nodes, y_nodes):
    """Thickness-averaged plane-stress (vp, vs, rho) of the Salvus mesh on an FD grid."""
    import h5py
    from scipy.interpolate import griddata

    from plane_stress import plane_stress_moduli

    with h5py.File(mesh_path(), "r") as f:
        c = f["MODEL/coordinates"][()].reshape(-1, 3)
        d = f["MODEL/data"][()]  # (element, [RHO, VP, VS], point)
    rho, vp, vs = (d[:, k, :].reshape(-1).astype(np.float64) for k in range(3))
    lam_ps, mu = plane_stress_moduli(vp, vs, rho)
    lam_ps, mu = np.asarray(lam_ps), np.asarray(mu)

    z_levels = np.unique(np.round(c[:, 2], 6))
    assert len(z_levels) == 5, f"expected one 4th-order element through thickness, got {z_levels}"
    w = GLL5_WEIGHTS[np.searchsorted(z_levels, np.round(c[:, 2], 6))]
    key = np.round(c[:, :2], 6)
    cols, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    def col_avg(v):
        return np.bincount(inverse, w * v) / np.bincount(inverse, w)

    lam_e, mu_e, rho_e = col_avg(lam_ps), col_avg(mu), col_avg(rho)
    X, Y = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    grid = lambda v: griddata(cols, v, (X, Y), method="linear")
    lam_g, mu_g, rho_g = grid(lam_e), grid(mu_e), grid(rho_e)
    # invert the plane-stress relation for an equivalent (vp, vs)
    lam3 = 2 * mu_g * lam_g / (2 * mu_g - lam_g)
    vs_g = np.sqrt(mu_g / rho_g)
    vp_g = np.sqrt((lam3 + 2 * mu_g) / rho_g)
    background = dict(vp=float(np.median(vp)), vs=float(np.median(vs)), rho=float(np.median(rho)))
    return vp_g.astype(np.float32), vs_g.astype(np.float32), rho_g.astype(np.float32), background


def load_salvus():
    from scipy.io import netcdf_file

    events = []
    for path in sorted(SALVUS_DIR.glob("event_*_velocity_*.nc")):
        with netcdf_file(path, "r", mmap=False) as nc:
            v = nc.variables["velocity"]
            comp = v._attributes["component"]
            comp = comp.decode() if isinstance(comp, bytes) else comp
            events.append(dict(
                name=path.stem.split("_velocity_")[0], component=comp,
                x=nc.variables["x"].data.copy(), y=nc.variables["y"].data.copy(),
                t=nc.variables["time"].data.copy(), v=np.nan_to_num(v.data.copy()),
            ))
    return events


def source_positions():
    """Salvus event order, as in the SH0 notebook: sides x0, y0, x1, y1; 3 per side."""
    side, inner = L / 2 - 0.2, np.linspace(-0.25, 0.25, 3)
    return np.array([[-side, y] for y in inner] + [[x, -side] for x in inner]
                    + [[side, y] for y in inner] + [[x, side] for x in inner])


# ---------------------------------------------------------------------------- metrics


def norm_traces(a):
    return a / np.maximum(np.max(np.abs(a), axis=0, keepdims=True), 1e-30)


def best_lags(pred, obs, dt, max_lag_s=100e-6):
    n = int(max_lag_s / dt)
    lags = np.arange(-n, n + 1)
    p, o = norm_traces(pred), norm_traces(obs)
    cc = np.stack([np.sum(np.roll(p, k, axis=0) * o, axis=0) for k in lags])
    return lags[np.argmax(cc, axis=0)] * dt  # >0: prediction must be delayed


def metrics(pred, obs, sectors, dt):
    p, o = norm_traces(pred), norm_traces(obs)
    out = {}
    for name, sel in sectors.items():
        if sel.sum() == 0:
            continue
        rel = np.linalg.norm(p[:, sel] - o[:, sel]) / np.linalg.norm(o[:, sel])
        corr = np.sum(p[:, sel] * o[:, sel], 0) / (np.linalg.norm(p[:, sel], axis=0) * np.linalg.norm(o[:, sel], axis=0))
        out[name] = dict(misfit=float(rel), median_corr=float(np.median(corr)),
                         median_lag_us=float(np.median(best_lags(pred[:, sel], obs[:, sel], dt)) * 1e6))
    return out


# ------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=1e-3)
    parser.add_argument("--cfl", type=float, default=0.6)
    parser.add_argument("--events", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--with-acoustic", action="store_true",
                        help="also score the SH0 notebook's 2-D acoustic set-up")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    from plane_stress import plane_stress_cpml

    f0, shift = load_wavelet()
    salvus = load_salvus()
    t_obs = salvus[0]["t"]
    dt_obs = float(t_obs[1] - t_obs[0])

    n = int(round(L / args.dx)) + 1
    dx = L / (n - 1)
    nodes = -L / 2 + np.arange(n) * dx
    vp, vs, rho, bg = effective_plane_stress_model(nodes, nodes)
    print(f"device={jax.devices()[0]}  grid {n}x{n} dx={dx*1e3:.2f} mm | wavelet: Ricker f0={f0:.0f} Hz, "
          f"shift={shift} s | background {bg} | effective inclusion min vs={vs.min():.0f} vp={vp.min():.0f}")
    dt = args.cfl * dx / (float(vp.max()) * np.sqrt(2.0))
    t_sim = t_obs[0] + np.arange(int(np.ceil((t_obs[-1] - t_obs[0]) / dt)) + 1) * dt
    stf = ricker(t_sim, f0, shift)
    pml = int(round(ABC_WIDTH / dx))
    print(f"dt={dt*1e6:.3f} us, steps={len(t_sim)}, t0={t_sim[0]*1e6:.1f} us, CPML {pml} cells")

    xr, yr = salvus[0]["x"], salvus[0]["y"]
    keep_x, keep_y = np.where(np.abs(xr) <= INTERIOR)[0], np.where(np.abs(yr) <= INTERIOR)[0]
    RX, RY = np.meshgrid(xr[keep_x], yr[keep_y], indexing="ij")
    rec_m = np.stack([RX.ravel(), RY.ravel()], 1)
    rec_i = (rec_m + L / 2) / dx

    homo = {k: np.full_like(v, bg[k]) for k, v in (("vp", vp), ("vs", vs), ("rho", rho))}
    models = {"plane_stress_inclusion": (vp, vs, rho),
              "plane_stress_homogeneous": (homo["vp"], homo["vs"], homo["rho"])}

    run = jax.jit(lambda vp_, vs_, rho_, s_i, fdir: plane_stress_cpml(
        vp_, vs_, rho_, s_i, fdir, jnp.asarray(stf), dx, dx, dt, jnp.asarray(rec_i), f0, pml_width=pml))

    src_m = source_positions()
    results, traces_out = [], {}
    for e in args.events:
        ev = salvus[e]
        comp = {"x": 0, "y": 1}[ev["component"]]
        fdir = np.array([0.0, 1.0]) if e // 3 in (0, 2) else np.array([1.0, 0.0])  # transverse force
        obs_full = ev["v"][np.ix_(keep_x, keep_y)].reshape(-1, len(t_obs)).T  # (nt, nrec)
        window = t_obs >= 0.0
        obs = obs_full[window]
        # sectors: SH0 = within 30 deg of the source->centre axis, S0 = 60-90 deg
        to_centre = -src_m[e] / np.linalg.norm(src_m[e])
        rel = rec_m - src_m[e]
        cosang = (rel @ to_centre) / np.linalg.norm(rel, axis=1)
        ang = np.degrees(np.arccos(np.clip(np.abs(cosang), 0, 1)))
        far = np.linalg.norm(rel, axis=1) > 0.1
        sectors = {"all": far, "SH0_axis": far & (ang < 30), "S0_broadside": far & (ang > 60)}

        record = {"event": ev["name"], "component": ev["component"], "source_m": src_m[e].tolist()}
        for name, (a, b, c) in models.items():
            t0 = time.perf_counter()
            tr = np.asarray(jax.block_until_ready(run(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c),
                                                      jnp.asarray((src_m[e] + L / 2) / dx), jnp.asarray(fdir))))
            elapsed = time.perf_counter() - t0
            pred_full = np.stack([np.interp(t_obs, t_sim, tr[:, r, comp]) for r in range(tr.shape[1])], 1)
            pred = pred_full[window]
            sign = np.sign(np.sum(pred * obs)) or 1.0
            record[name] = dict(metrics(sign * pred, obs, sectors, dt_obs), sign=float(sign), solve_s=elapsed)
            traces_out[f"{ev['name']}_{name}"] = (sign * pred_full).astype(np.float32)
        if args.with_acoustic:
            record["acoustic_notebook"] = acoustic_notebook_metrics(e, rec_m, obs, sectors, t_obs, window, dt_obs, traces_out, ev)
        results.append(record)
        print(f"{ev['name']} v_{ev['component']}: " + " | ".join(
            f"{k}: misfit all={v['all']['misfit']:.3f} SH0={v['SH0_axis']['misfit']:.3f} "
            f"S0={v['S0_broadside']['misfit']:.3f} lag={v['all']['median_lag_us']:+.1f}us"
            for k, v in record.items() if isinstance(v, dict)), flush=True)

    tag = f"dx{dx*1e3:.2f}mm"
    np.savez_compressed(RESULTS / f"traces_{tag}.npz", t=t_obs, rec_m=rec_m, **traces_out,
                        **{f"{salvus[e]['name']}_salvus": salvus[e]["v"][np.ix_(keep_x, keep_y)].reshape(-1, len(t_obs)).T
                           for e in args.events})
    (RESULTS / f"summary_{tag}.json").write_text(json.dumps(
        {"dx": dx, "dt": dt, "wavelet": {"f0": f0, "shift": shift}, "background": bg,
         "results": results}, indent=2))


def acoustic_notebook_metrics(e, rec_m, obs, sectors, t_obs, window, dt_obs, traces_out, ev):
    """The SH0 notebook's 2-D acoustic model (c=2100, its inclusion), its time axis (JAX t=0 = Salvus t=0)."""
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, str(SCRIPTS))
    from run_oed_multiseed import Simulation, scientific_config

    global _ACOUSTIC
    if "_ACOUSTIC" not in globals():
        _ACOUSTIC = Simulation(scientific_config())
    sim = _ACOUSTIC
    src_i = jnp.asarray(sim.xy_to_indices(source_positions()[e]), jnp.float32)
    rec_i = jnp.asarray(sim.xy_to_indices(rec_m), jnp.float32)
    n_steps = int(np.ceil(t_obs[-1] / sim.dt))
    tr = sim.forward_solver(sim.true_model, sim.density_grid, src_i, sim.config.f0, sim.dx, sim.dy, sim.dt,
                            n_steps, receiver_is=rec_i, output_wavefield=False, pml_width=sim.pml_width,
                            R_coeff=sim.config.pml_reflection_coefficient, m=sim.config.pml_polynomial_order,
                            kappa_max=sim.config.pml_kappa_max, alpha_max=None)[0]
    tr = np.asarray(tr)
    t_jax = np.arange(n_steps) * sim.dt
    pred_full = np.stack([np.interp(t_obs, t_jax, tr[:, r], left=0.0) for r in range(tr.shape[1])], 1)
    pred = pred_full[window]
    sign = np.sign(np.sum(pred * obs)) or 1.0
    traces_out[f"{ev['name']}_acoustic_notebook"] = (sign * pred_full).astype(np.float32)
    return dict(metrics(sign * pred, obs, sectors, dt_obs), sign=float(sign))


if __name__ == "__main__":
    main()

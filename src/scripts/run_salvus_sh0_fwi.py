#!/usr/bin/env python3
"""Run SH0 Salvus-to-acoustic FWI from a standalone Python script.

This script is a stripped-down, server-oriented version of the inversion part
of ``test_salvus_forward_solve_dispersion_SH0.ipynb``. It keeps only:

* centered-domain geometry setup,
* Salvus trace loading and interpolation onto the acoustic time axis,
* trace-wise normalization,
* acoustic CPML forward modeling,
* stochastic Adam FWI updates,
* result serialization.

The acoustic solver still receives source/receiver positions as grid indices,
but geometry is specified in centered meters to match the Salvus convention.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    """Physical and acquisition setup in meters."""

    lx: float = 1.1
    ly: float = 1.1
    c_min: float = 1300.0
    c_max: float = 1900.0
    density: float = 2700.0
    scatterer_center_x_m: float = -0.05
    scatterer_center_y_m: float = 0.05
    scatterer_side_m: float = 0.055
    f0_max: float = 25e3
    ppw: float = 8.0
    pml_thickness_m: float = 0.1
    src_edge_offset_m: float = 0.2
    src_inner_min_m: float = -0.25
    src_inner_max_m: float = 0.25
    n_src_per_side: int = 3
    rec_min_m: float = -0.15
    rec_max_m: float = 0.15
    n_rx: int = 30
    n_ry: int = 30
    total_time_s: float = 500e-6
    f0: float = 25e3


@dataclass(frozen=True)
class InversionConfig:
    """Optimizer and execution setup."""

    num_steps: int = 5000
    learning_rate: float = 25.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.9
    seed: int = 42
    source_mask_radius_grid_points: int = 5
    smooth_iterations: int = 1
    display_every: int = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone SH0 FWI using Salvus traces as data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("all_traces_SH0_scat_absor.npz"),
        help="Path to Salvus npz file with keys 'all_traces' and 't'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fwi_results_SH0_server.npz"),
        help="Output npz path for inverted model and diagnostics.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of stochastic FWI update steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=25.0,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="JAX PRNG seed.",
    )
    parser.add_argument(
        "--display-every",
        type=int,
        default=10,
        help="Progress-bar update interval.",
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "gpu"),
        default="gpu",
        help="JAX platform. Use 'cpu' for debugging or machines without GPU.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Use a fully JIT-scanned loop instead of a Python tqdm loop.",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    return args


def resolve_data_path(path: Path) -> Path:
    """Resolve data path from cwd first, then from this script's directory."""

    if path.exists() or path.is_absolute():
        return path

    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative

    return path


def configure_jax(platform: str):
    """Import and configure JAX after CLI parsing."""

    if platform == "gpu":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax

    jax.config.update("jax_platform_name", platform)

    import jax.numpy as jnp
    from jax import jit, random, value_and_grad
    from jax.example_libraries import optimizers
    from jaxdf.operators import compose
    from jwave.signal_processing import smooth
    from tqdm.auto import trange

    from hybridoed.forward import acoustic2D_cpml_minmem

    return {
        "jax": jax,
        "jnp": jnp,
        "jit": jit,
        "random": random,
        "value_and_grad": value_and_grad,
        "optimizers": optimizers,
        "compose": compose,
        "smooth": smooth,
        "trange": trange,
        "acoustic2D_cpml_minmem": acoustic2D_cpml_minmem,
    }


def build_setup(geo: GeometryConfig, jnp):
    """Build model, source, receiver, and time-step arrays."""

    fmax = 2.0 * geo.f0_max
    dx_req = geo.c_min / (geo.ppw * fmax)
    dx = dy = dx_req

    nx = int(np.floor(geo.lx / dx)) + 1
    ny = int(np.floor(geo.ly / dy)) + 1
    dx = geo.lx / (nx - 1)
    dy = geo.ly / (ny - 1)

    x_min_m, x_max_m = -geo.lx / 2, geo.lx / 2
    y_min_m, y_max_m = -geo.ly / 2, geo.ly / 2

    def x_to_i(x_m):
        return (np.asarray(x_m) - x_min_m) / dx

    def y_to_i(y_m):
        return (np.asarray(y_m) - y_min_m) / dy

    def xy_to_indices(xy_m):
        xy_m = np.asarray(xy_m, dtype=float)
        return np.stack([x_to_i(xy_m[..., 0]), y_to_i(xy_m[..., 1])], axis=-1)

    dt_max = min(dx, dy) / (geo.c_max * np.sqrt(2.0))
    dt = 0.8 * dt_max
    pml_width = max(4, int(round(geo.pml_thickness_m / dx)))

    true_model = jnp.full((nx, ny), geo.c_max, dtype=jnp.float32)
    homo_model = jnp.full((nx, ny), geo.c_max, dtype=jnp.float32)

    half_side = geo.scatterer_side_m / 2
    scx0 = geo.scatterer_center_x_m - half_side
    scx1 = geo.scatterer_center_x_m + half_side
    scy0 = geo.scatterer_center_y_m - half_side
    scy1 = geo.scatterer_center_y_m + half_side
    ix0, ix1 = int(x_to_i(scx0)), int(x_to_i(scx1))
    iy0, iy1 = int(y_to_i(scy0)), int(y_to_i(scy1))
    true_model = true_model.at[ix0:ix1, iy0:iy1].set(geo.c_min)

    density_grid_homo = jnp.full_like(homo_model, geo.density)

    src_side_m = geo.lx / 2 - geo.src_edge_offset_m
    src_inner_m = np.linspace(
        geo.src_inner_min_m, geo.src_inner_max_m, geo.n_src_per_side
    )
    src_positions_m = np.array(
        [[-src_side_m, y] for y in src_inner_m]
        + [[x, -src_side_m] for x in src_inner_m]
        + [[src_side_m, y] for y in src_inner_m]
        + [[x, src_side_m] for x in src_inner_m],
        dtype=float,
    )
    src_indices = jnp.array(xy_to_indices(src_positions_m), dtype=jnp.float32)

    rec_x_m = np.linspace(geo.rec_min_m, geo.rec_max_m, geo.n_rx)
    rec_y_m = np.linspace(geo.rec_min_m, geo.rec_max_m, geo.n_ry)
    rec_xx, rec_yy = np.meshgrid(rec_x_m, rec_y_m, indexing="ij")
    receiver_positions_m = np.stack([rec_xx.ravel(), rec_yy.ravel()], axis=1)
    receiver_is = jnp.array(xy_to_indices(receiver_positions_m), dtype=jnp.float32)

    n_steps = int(np.ceil(geo.total_time_s / dt))
    time = jnp.arange(n_steps) * dt

    return {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "dt": dt,
        "pml_width": pml_width,
        "true_model": true_model,
        "homo_model": homo_model,
        "density_grid_homo": density_grid_homo,
        "src_indices": src_indices,
        "src_positions_m": src_positions_m,
        "receiver_is": receiver_is,
        "receiver_positions_m": receiver_positions_m,
        "n_steps": n_steps,
        "time": time,
        "scatterer_bounds_m": np.array([scx0, scx1, scy0, scy1]),
    }


def interpolate_traces_to_time(traces: np.ndarray, trace_time: np.ndarray, target_time):
    """Interpolate traces from Salvus time samples to acoustic solver samples."""

    target_time = np.asarray(target_time, dtype=float)
    traces = np.asarray(traces, dtype=float)
    trace_time = np.asarray(trace_time, dtype=float)

    if traces.shape[-1] == target_time.shape[-1]:
        return traces.astype(np.float32, copy=False)
    if trace_time.shape[-1] != traces.shape[-1]:
        trace_time = np.linspace(trace_time[0], trace_time[-1], traces.shape[-1])

    out = np.empty(traces.shape[:-1] + (target_time.shape[-1],), dtype=np.float32)
    flat_in = traces.reshape(-1, traces.shape[-1])
    flat_out = out.reshape(-1, target_time.shape[-1])
    for i, trace in enumerate(flat_in):
        flat_out[i] = np.interp(target_time, trace_time, trace)
    return out


def load_salvus_data(data_path: Path, target_time, jnp):
    """Load, time-align, and trace-normalize Salvus data."""

    salvus_out = np.load(data_path)
    traces = salvus_out["all_traces"]
    salvus_time = salvus_out["t"]

    traces = interpolate_traces_to_time(traces, salvus_time, target_time)
    traces = jnp.asarray(traces, dtype=jnp.float32)
    scale = jnp.maximum(jnp.max(jnp.abs(traces), axis=2, keepdims=True), 1e-12)
    return traces / scale, salvus_time


def build_source_mask(shape, src_indices, radius, jnp):
    """Mask a small square around each source to suppress source artifacts."""

    mask = jnp.ones(shape, dtype=jnp.float32)
    nx, ny = shape
    for src in range(src_indices.shape[0]):
        src_x_idx = int(src_indices[src, 0])
        src_y_idx = int(src_indices[src, 1])
        x_min = max(src_x_idx - radius, 0)
        x_max = min(src_x_idx + radius + 1, nx)
        y_min = max(src_y_idx - radius, 0)
        y_max = min(src_y_idx + radius + 1, ny)
        mask = mask.at[x_min:x_max, y_min:y_max].set(0.0)
    return mask


def run_fwi(
    setup,
    salvus_data,
    mask,
    geo: GeometryConfig,
    inv: InversionConfig,
    jax_modules,
    use_progress: bool,
):
    """Run stochastic single-source FWI."""

    jax = jax_modules["jax"]
    jnp = jax_modules["jnp"]
    jit = jax_modules["jit"]
    random = jax_modules["random"]
    value_and_grad = jax_modules["value_and_grad"]
    optimizers = jax_modules["optimizers"]
    compose = jax_modules["compose"]
    smooth = jax_modules["smooth"]
    trange = jax_modules["trange"]
    acoustic2D_cpml_minmem = jax_modules["acoustic2D_cpml_minmem"]

    homo_model = setup["homo_model"]
    density_grid_homo = setup["density_grid_homo"]
    src_indices = setup["src_indices"]
    receiver_is = setup["receiver_is"]

    def get_sound_speed(params):
        return params + compose(params)(jax.nn.sigmoid)

    def smooth_gradient(gradient):
        x = gradient * mask
        for _ in range(inv.smooth_iterations):
            x = smooth(x)
        return x

    init_fun, update_fun, get_params = optimizers.adam(
        inv.learning_rate, inv.adam_beta1, inv.adam_beta2
    )
    opt_state = init_fun(homo_model)
    available_src = jnp.arange(src_indices.shape[0])
    key = random.PRNGKey(inv.seed)
    key, _ = random.split(key)

    def loss_func(params, src_num):
        p_rec = acoustic2D_cpml_minmem(
            get_sound_speed(params),
            density_grid_homo,
            src_indices[src_num],
            geo.f0,
            setup["dx"],
            setup["dy"],
            setup["dt"],
            setup["n_steps"],
            receiver_is=receiver_is,
            output_wavefield=True,
            pml_width=setup["pml_width"],
            R_coeff=1e-6,
            m=3,
            kappa_max=3.0,
            alpha_max=None,
        )

        pred = p_rec[0].T
        pred = pred / jnp.maximum(jnp.max(jnp.abs(pred), axis=1, keepdims=True), 1e-12)
        data = salvus_data[src_num]
        return jnp.mean((pred - data) ** 2)

    loss_with_grad = value_and_grad(loss_func, argnums=0)

    @jit
    def update_step(opt_state, key, step_idx):
        params = get_params(opt_state)
        key, subkey = random.split(key)
        src_num = random.choice(subkey, available_src)
        loss_value, gradient = loss_with_grad(params, src_num)
        gradient = smooth_gradient(gradient)
        opt_state = update_fun(step_idx, gradient, opt_state)
        return opt_state, key, loss_value, src_num

    if use_progress:
        losses = []
        chosen_sources = []

        opt_state, key, loss_value, src_num = update_step(opt_state, key, 0)
        losses.append(float(loss_value.block_until_ready()))
        chosen_sources.append(int(src_num))

        pbar = trange(1, inv.num_steps, desc="FWI", dynamic_ncols=True)
        for step_idx in pbar:
            opt_state, key, loss_value, src_num = update_step(opt_state, key, step_idx)
            if step_idx % inv.display_every == 0:
                loss_scalar = float(loss_value.block_until_ready())
                pbar.set_postfix(loss=f"{loss_scalar:.3e}", src=int(src_num))
            else:
                loss_scalar = float(loss_value)
            losses.append(loss_scalar)
            chosen_sources.append(int(src_num))

        final_model = get_params(opt_state)
        return final_model, jnp.asarray(losses), jnp.asarray(chosen_sources)

    @jit
    def run_scan(opt_state, key):
        def body(carry, step_idx):
            opt_state, key = carry
            opt_state, key, loss_value, src_num = update_step(opt_state, key, step_idx)
            return (opt_state, key), (loss_value, src_num)

        (opt_state, key), (losses, chosen_sources) = jax.lax.scan(
            body, (opt_state, key), jnp.arange(inv.num_steps)
        )
        return opt_state, losses, chosen_sources

    opt_state, losses, chosen_sources = run_scan(opt_state, key)
    final_model = get_params(opt_state)
    return final_model, losses, chosen_sources


def save_results(output_path: Path, setup, final_model, losses, chosen_sources, geo, inv, jax):
    """Write model, diagnostics, and geometry metadata to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        final_model=np.asarray(jax.device_get(final_model)),
        true_model=np.asarray(jax.device_get(setup["true_model"])),
        initial_model=np.asarray(jax.device_get(setup["homo_model"])),
        losses=np.asarray(jax.device_get(losses)),
        chosen_sources=np.asarray(jax.device_get(chosen_sources)),
        src_indices=np.asarray(jax.device_get(setup["src_indices"])),
        src_positions_m=setup["src_positions_m"],
        receiver_indices=np.asarray(jax.device_get(setup["receiver_is"])),
        receiver_positions_m=setup["receiver_positions_m"],
        time=np.asarray(jax.device_get(setup["time"])),
        dx=setup["dx"],
        dy=setup["dy"],
        dt=setup["dt"],
        pml_width=setup["pml_width"],
        scatterer_bounds_m=setup["scatterer_bounds_m"],
        geometry_config=np.array(str(asdict(geo))),
        inversion_config=np.array(str(asdict(inv))),
    )


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    jax_modules = configure_jax(args.platform)
    jax = jax_modules["jax"]
    jnp = jax_modules["jnp"]

    geo = GeometryConfig()
    inv = InversionConfig(
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        display_every=args.display_every,
    )

    setup = build_setup(geo, jnp)
    print(
        "Grid: "
        f"nx={setup['nx']}, ny={setup['ny']}, "
        f"dx={setup['dx'] * 1e3:.2f} mm, dt={setup['dt'] * 1e6:.2f} us, "
        f"n_steps={setup['n_steps']}"
    )
    print(f"Sources: {setup['src_indices'].shape[0]}, receivers: {setup['receiver_is'].shape[0]}")

    salvus_data, salvus_time = load_salvus_data(data_path, setup["time"], jnp)
    print(
        "Loaded Salvus data: "
        f"{tuple(salvus_data.shape)} from {data_path} "
        f"(original nt={salvus_time.shape[0]})"
    )

    mask = build_source_mask(
        setup["homo_model"].shape,
        setup["src_indices"],
        inv.source_mask_radius_grid_points,
        jnp,
    )

    final_model, losses, chosen_sources = run_fwi(
        setup,
        salvus_data,
        mask,
        geo,
        inv,
        jax_modules,
        use_progress=not args.no_progress,
    )
    final_model.block_until_ready()

    save_results(args.output, setup, final_model, losses, chosen_sources, geo, inv, jax)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()

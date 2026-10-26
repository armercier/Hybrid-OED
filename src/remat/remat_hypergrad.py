"""One outer hypergradient of the full-autodiff OED objectives, with or without remat.

Computes the receiver gradient of the oracle or waveform-misfit objective for one
design round (K unrolled SGD FWI updates), exactly as ``outer_step`` in
``run_oed_multiseed_full_autodiff.py`` does, and reports loss, gradient, time and
peak memory.  The paper code is imported unchanged.  ``--remat step`` swaps in
``RematSimulation``, whose only difference is that each unrolled FWI update is
wrapped in ``jax.checkpoint``: its intermediates (the forward and adjoint wave
simulations) are recomputed during the backward pass instead of being stored.

Usage (one configuration per process, so peak memory is isolated):
  python src/remat/remat_hypergrad.py --remat step --method oracle --device gpu --updates 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "src" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_oed_multiseed import configure_jax  # noqa: E402

RESULTS = REPO / "remat_results"


def make_remat_class(probe_batch: int = 4):
    """Subclass of the paper ``Simulation`` with rematerialised FWI updates and illumination."""
    from run_oed_multiseed_full_autodiff import Simulation

    class RematSimulation(Simulation):
        def fwi_core(self, initial_model, receiver_indices, source_sequence, observed_data):
            """The paper's ``fwi_core`` with each SGD update rematerialised."""
            jnp = self.jnp
            source_sequence = jnp.asarray(source_sequence, dtype=jnp.int32)
            observed_data = jnp.asarray(observed_data)
            init_fun, update_fun, get_params = self.optimizers.sgd(
                self.config.fwi_learning_rate
            )
            initial_state = init_fun(initial_model)

            def loss_function(parameters, source_number):
                prediction = self.simulate_normalized_receivers(
                    self.get_sound_speed(parameters),
                    self.source_indices[source_number],
                    receiver_indices,
                )
                return jnp.mean((prediction - observed_data[source_number]) ** 2)

            loss_with_gradient = self.value_and_grad(loss_function, argnums=0)

            # Only the carried SGD state is stored per update; the wave simulations
            # inside are recomputed when the outer backward pass reaches them.
            @self.jax.checkpoint
            def body(state, inputs):
                step_index, source_number = inputs
                parameters = get_params(state)
                loss, gradient = loss_with_gradient(parameters, source_number)
                state = update_fun(step_index, self.smooth_gradient(gradient), state)
                return state, loss

            steps = jnp.arange(source_sequence.shape[0], dtype=jnp.int32)
            final_state, losses = self.lax.scan(body, initial_state, (steps, source_sequence))
            return get_params(final_state), losses

        def illumination_loss(self, model_parameters, receiver_indices, probe_seed):
            """The paper's illumination loss with sources and probes rematerialised.

            Identical probes and averaging; the paper ``vmap``s over all sources
            and probes at once, here sources run one at a time and probes in
            batches of ``probe_batch``, each recomputed in the backward pass.
            """
            jnp, random = self.jnp, self.jax.random
            base_key = random.PRNGKey(jnp.asarray(probe_seed, dtype=jnp.uint32))
            sound_speed = self.get_sound_speed(model_parameters)

            @self.jax.checkpoint
            def one_source(source_number):
                source = self.source_indices[source_number]

                def forward(model):
                    return self.forward_solver(
                        model,
                        self.density_grid_homo,
                        source,
                        self.config.f0,
                        self.dx,
                        self.dy,
                        self.dt,
                        self.n_steps,
                        receiver_is=receiver_indices,
                        output_wavefield=False,
                        pml_width=self.pml_width,
                        R_coeff=self.config.pml_reflection_coefficient,
                        m=self.config.pml_polynomial_order,
                        kappa_max=self.config.pml_kappa_max,
                        alpha_max=None,
                    )[0]

                output, pullback = self.jax.vjp(forward, sound_speed)
                keys = random.split(
                    random.fold_in(base_key, source_number),
                    self.config.illumination_probes,
                )
                probes = self.vmap(
                    lambda key: random.rademacher(key, output.shape).astype(output.dtype)
                )(keys)
                probes = probes / jnp.sqrt(output.size)
                sensitivity = self.jax.checkpoint(lambda probe: pullback(probe)[0])
                sensitivities = self.lax.map(sensitivity, probes, batch_size=probe_batch)
                return jnp.mean(sensitivities**2, axis=0)

            source_numbers = jnp.arange(self.source_indices.shape[0], dtype=jnp.int32)
            illumination = jnp.mean(self.lax.map(one_source, source_numbers), axis=0)
            illumination = self.smooth(illumination)
            mean_illumination = jnp.mean(illumination)
            normalized = illumination / (mean_illumination + self.config.illumination_eps)
            return jnp.var(normalized) - self.config.illumination_alpha_strength * jnp.log(
                mean_illumination + self.config.illumination_eps
            )

    return RematSimulation


def build_simulation(remat: str, probe_batch: int = 4):
    from run_oed_multiseed_full_autodiff import Simulation, scientific_config

    cls = Simulation if remat == "none" else make_remat_class(probe_batch)
    return cls(scientific_config())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remat", choices=("none", "step"), required=True)
    parser.add_argument("--method", choices=("oracle", "waveform", "illumination"), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--probe-batch", type=int, default=4)
    args = parser.parse_args()

    configure_jax("cuda" if args.device == "gpu" else args.device)
    import jax
    import jax.numpy as jnp

    from hybridoed.oed_multiseed import make_fwi_source_sequence
    from run_oed_multiseed_full_autodiff import PeakRSSMonitor, make_outer_objective

    tag = f"{args.method}_{args.remat}_{args.device}_K{args.updates}"
    if args.method == "illumination":
        tag = f"illumination_{args.remat}_{args.device}_pb{args.probe_batch}"
    out_dir = RESULTS / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict = {"tag": tag, **vars(args), "platform": str(jax.devices()[0])}

    sim = build_simulation(args.remat, args.probe_batch)
    objective = make_outer_objective(sim, args.method, probe_seed=0)
    receivers = sim.clamp_receivers(jnp.asarray(sim.base_receiver_grid))
    sequence = jnp.asarray(
        make_fwi_source_sequence(args.seed, args.updates, int(sim.source_indices.shape[0])),
        dtype=jnp.int32,
    )
    grad_fn = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))

    t0 = time.perf_counter()
    compiled = grad_fn.lower(receivers, sim.homo_model, sequence).compile()
    record["compile_s"] = time.perf_counter() - t0
    analysis = compiled.memory_analysis()
    if analysis is not None:
        record["xla_temp_gib"] = analysis.temp_size_in_bytes / 2**30
        record["xla_total_gib"] = (analysis.temp_size_in_bytes + analysis.argument_size_in_bytes
                                   + analysis.output_size_in_bytes) / 2**30

    if not args.compile_only:
        try:
            with PeakRSSMonitor() as monitor:
                times = []
                for _ in range(2):
                    t0 = time.perf_counter()
                    (loss, _aux), gradient = compiled(receivers, sim.homo_model, sequence)
                    gradient.block_until_ready()
                    times.append(time.perf_counter() - t0)
            record.update(
                loss=float(loss), run_s=times, grad_norm=float(jnp.linalg.norm(gradient)),
                peak_rss_increase_gib=monitor.metrics()["peak_rss_increase_mb"] / 1024,
            )
            if args.device == "gpu":
                stats = jax.devices()[0].memory_stats() or {}
                record["gpu_peak_gib"] = stats.get("peak_bytes_in_use", 0) / 2**30
            np.save(out_dir / f"{tag}_grad.npy", np.asarray(gradient))
        except Exception as error:  # e.g. RESOURCE_EXHAUSTED on the 11 GB GPU
            record["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"

    (out_dir / f"{tag}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()

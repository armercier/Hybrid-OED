#!/usr/bin/env python3

import os

###############################################################################
# Decide on CPU or GPU here
use_gpu = False  # Set to False if you want CPU only
###############################################################################

if use_gpu:
    # Prevent JAX from preallocating most of the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # Force JAX to use GPU
    import jax
    jax.config.update("jax_platform_name", "gpu")
else:
    # Force JAX to use CPU
    import jax
    jax.config.update("jax_platform_name", "cpu")

# Now import JAX and the rest of your libraries
from jax import jit, vjp, vmap, pmap, random, value_and_grad, nn, profiler
import jax.numpy as jnp
import equinox as eqx
import optax
from matplotlib import pyplot as plt
from jaxdf.operators import compose
from jax.profiler import save_device_memory_profile

# Continue with your other imports...
import numpy as np
from jwave.geometry import Domain, Medium
from jwave.utils import display_complex_field
from hybridoed.forward import create_src_field, generate_2D_gridded_src_rec_positions
from hybridoed.oed import *
from functools import partial
from jax.example_libraries import optimizers
from tqdm import tqdm
from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import (
    Domain,
    Medium,
    Sensors,
    Sources,
    TimeAxis,
    circ_mask,
    points_on_circle,
)
from jwave.signal_processing import apply_ramp, gaussian_window, smooth, analytic_signal
from jaxdf.operators import gradient, functional

# ------------------------------------------------------------------------------
# Everything below is your original code, with no changes except where device usage
# might matter (e.g., in memory profiles or pushing/pulling data to a device).
# ------------------------------------------------------------------------------

key = random.PRNGKey(42)  # Random seed

# Load model
model = jnp.load('model1.npy')
print(model.shape)
model_index = 7
true_model = model[model_index,0,:,:]
plt.imshow(true_model)
plt.colorbar()
plt.show()
print(type(true_model))

from scipy.ndimage import gaussian_filter

# Apply a Gaussian blur
blurred_model = gaussian_filter(true_model, sigma=20)

# Visualize the blurred model
plt.figure()
plt.imshow(blurred_model, cmap='viridis')
plt.colorbar()
plt.title('Blurred Model')
plt.show()
print(blurred_model.shape)

# Homogeneous model with the average velocity
homogenous_model = jnp.ones(true_model.shape) * jnp.mean(true_model)

N = (70, 70)  # Grid size
dx = (1.0, 1.0)  # Spatial resolution
cfl=0.75

# Defining the domain
domain = Domain(N, dx)

source_freq = 200
source_mag = 1.3e-1
medium = Medium(domain=domain, sound_speed=true_model, density=1000., pml_size=10)

# Time axis
time_axis = TimeAxis.from_medium(medium, cfl=cfl)
t = time_axis.to_array()
source_mag = source_mag / time_axis.dt

s1 = source_mag * jnp.sin(2 * jnp.pi * source_freq * t + 100)
signal = gaussian_window(apply_ramp(s1, time_axis.dt, source_freq), t, 0.5e-2, 1e-2)

def ricker_wavelet(t, f, t_shift=0.0):
    """
    Generate a Ricker wavelet with a time shift.
    """
    t_shifted = t - t_shift  # Apply the time shift
    pi2 = (jnp.pi ** 2)
    a = (pi2 * f ** 2) * (t_shifted ** 2)
    wavelet = (1 - 2 * a) * jnp.exp(-a)
    return wavelet

t_shift = 0.005  # Time shift in seconds
signal = source_mag * ricker_wavelet(t, source_freq, t_shift)

src_coords_list, receiver_coords_list = generate_2D_gridded_src_rec_positions(
    N=(70, 70), num_sources=5, num_receivers=5
)
num_sources = src_coords_list.shape[0]
sensors_positions = (receiver_coords_list[:,0],receiver_coords_list[:,1])
sensors = Sensors(positions=sensors_positions)
source_positions = (src_coords_list[:,0],src_coords_list[:,1])

# Show the setup
fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 2]})
ax[0].imshow(medium.sound_speed.on_grid, cmap="gray")
ax[0].scatter(source_positions[1], source_positions[0], c="r", marker="x", label="sources")
ax[0].scatter(sensors_positions[1], sensors_positions[0], c="g", marker=".", label="sensors")
ax[0].legend(loc="lower right")
ax[0].set_title("Sound speed")
ax[0].axis("off")
ax[1].plot(signal, label="Source 1", c="k")
ax[1].set_title("Source signals")

fcn_params = {
    "criterion_threshold": 1e-10,
    "regularisation_loss": 0.0,
    "norm_loss": 0.0,
    "top_k_loss": 0.0,
    "differentiable_mask_sharpness": 10.0,
    "number_of_k":10,
    "hidden_size": 56,
    "num_hidden_layers": 3,
    "learning_rate": 1e-3,
    "num_iterations": 15,
    "print_gradients": False,
    "num_sources": num_sources,
}

################################################################################
# Single-source simulation
################################################################################
from jax import lax

src_signal = jnp.stack([signal])

@jit
def single_source_simulation(sound_speed, source_num):
    x = lax.dynamic_slice(source_positions[0], (source_num,), (1,))
    y = lax.dynamic_slice(source_positions[1], (source_num,), (1,))
    sources = Sources((x, y), src_signal, dt=time_axis.dt, domain=domain)
    medium_local = Medium(domain=domain, sound_speed=sound_speed, pml_size=10)
    rf_signals = simulate_wave_propagation(
        medium_local, time_axis, sources=sources, sensors=sensors, checkpoint=True
    )
    return rf_signals[..., 0]

p = single_source_simulation(medium.sound_speed, num_sources // 2)
plt.figure(figsize=(6, 4.5))
plt.imshow(p, cmap="RdBu_r", interpolation="nearest", aspect="auto")
plt.colorbar()
plt.title("Acoustic traces")
plt.xlabel("Sensor index")
plt.ylabel("Time")
plt.show()

mask = jnp.ones(domain.N)
mask = mask.at[sensors_positions[0], sensors_positions[1]].set(0.0)

from jaxdf.operators import compose

def get_sound_speed(params):
    return params + compose(params)(nn.sigmoid) * mask

################################################################################
# Fully Connected Neural Network
################################################################################
class FullyConnectedNN(eqx.Module):
    layers: list
    activations: list

    def __init__(self, input_size, hidden_size, num_hidden_layers, key):
        keys = jax.random.split(key, num_hidden_layers + 1)
        # Build layers
        self.layers = [eqx.nn.Linear(input_size, hidden_size, key=keys[0])] + \
                      [eqx.nn.Linear(hidden_size, hidden_size, key=k) for k in keys[1:-1]] + \
                      [eqx.nn.Linear(hidden_size, input_size, key=keys[-1])]
        # Activation
        self.activations = [jax.nn.tanh] * num_hidden_layers + [jax.nn.sigmoid]

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x

def differentiable_mask(probabilities, sharpness=10.0):
    """Convert probabilities to a 'soft' binary mask."""
    return jax.nn.sigmoid(sharpness * (probabilities - 0.5))

# Simple FWI with weighting by probabilities
from jax import value_and_grad

def fwi(probabilities):
    """
    Run your FWI (inner) optimization using `probabilities` as weights for each source.
    Return final model parameters (or final reconstruction) and the final data mismatch.
    """
    # Just a small example of an FWI routine:
    init_params = blurred_model  # Start from blurred model
    init_fun, update_fun, get_params = optimizers.adam(20.0, b1=0.9, b2=0.9)
    opt_state = init_fun(init_params)

    # Precompute data for all sources (the "true" data)
    batch_simulations = vmap(single_source_simulation, in_axes=(None, 0))
    true_data = batch_simulations(medium.sound_speed, jnp.arange(num_sources))

    def single_fwi_loss(params, src_idx):
        c0 = get_sound_speed(params)
        predicted = single_source_simulation(c0, src_idx)
        # L2 mismatch:
        return jnp.mean((predicted - true_data[src_idx]) ** 2)

    loss_with_grad = value_and_grad(single_fwi_loss, argnums=0)

    @jit
    def update_step(opt_state, key, step, probabilities):
        params_local = get_params(opt_state)
        # randomly pick a source
        src_idx = random.choice(key, jnp.arange(num_sources))
        # Weighted gradient:
        loss_val, grad = loss_with_grad(params_local, src_idx)
        grad = grad * probabilities[src_idx]
        return loss_val, update_fun(step, grad, opt_state)

    # Main optimization loop for the FWI
    loop_range = 55
    kkey = random.PRNGKey(123)
    for k in range(loop_range):
        kkey, subkey = random.split(kkey)
        lossval, opt_state = update_step(opt_state, subkey, k, probabilities)

    final_params = get_params(opt_state)
    return final_params, lossval

################################################################################
# Outer-level differentiable objective
################################################################################
def differentiable_loss_fn(model, x, criterion_threshold, sharpness=10.0, mask_penalty=0.1):
    """
    1. Model outputs a set of probabilities (one per source).
    2. Convert to soft mask (differentiable).
    3. Run FWI with that mask weighting.
    4. Compare final model to ground truth or measure final FWI mismatch.
    5. Add small regularizers on the mask to encourage discrete usage.
    """
    probabilities = model(x)  
    soft_mask = differentiable_mask(probabilities, sharpness)

    # Simple regularization (encourage probabilities to be near 0 or 1):
    regularization_loss = mask_penalty * jnp.mean(soft_mask * (1 - soft_mask))
    norm_loss = jnp.linalg.norm(soft_mask, ord=1)

    # Run FWI with these probabilities
    inverted_model, fwi_loss = fwi(probabilities)

    # You can measure how close the inverted model is to the true model:
    # e.g., L1 difference or L2 difference:
    final_misfit = jnp.linalg.norm(inverted_model - true_model, ord=1)

    # Combine everything into a single scalar objective:
    total_loss = final_misfit + \
                 fcn_params["regularisation_loss"] * regularization_loss + \
                 fcn_params["norm_loss"] * norm_loss

    return total_loss

################################################################################
# Training loop for the outer (probability) model
################################################################################
def train_step(model, criterion_threshold, optimizer, opt_state, x, logs):
    loss_and_grad_fn = eqx.filter_value_and_grad(differentiable_loss_fn)
    loss, grads = loss_and_grad_fn(
        model, x,
        criterion_threshold=criterion_threshold,
        sharpness=fcn_params["differentiable_mask_sharpness"],
        mask_penalty=1.0
    )

    if fcn_params["print_gradients"]:
        # Print gradient stats (optional)
        jax.tree_util.tree_map(lambda g: print(f"Grad shape: {g.shape}, min={g.min()}, max={g.max()}"), grads)

    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)

    probabilities = model(x)
    soft_mask = differentiable_mask(probabilities, fcn_params["differentiable_mask_sharpness"])

    # Store logs
    logs["probabilities"].append(probabilities)
    logs["masks"].append(soft_mask)

    return loss, model, opt_state

################################################################################
# Actually run the training
################################################################################
def main():
    global model
    global p_data

    # Build the network
    key = jax.random.PRNGKey(42)
    input_size = fcn_params["num_sources"]
    hidden_size = fcn_params["hidden_size"]
    num_hidden_layers = fcn_params["num_hidden_layers"]

    # model is your NN that will learn the probability vector
    model = FullyConnectedNN(input_size, hidden_size, num_hidden_layers, key)
    optimizer = optax.adamw(fcn_params["learning_rate"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # x is just a placeholder input to the network. In your code, you used ones()
    x = jnp.ones(input_size)

    losses = []
    logs = {"probabilities": [], "masks": []}
    for step in range(fcn_params["num_iterations"]):
        loss, model, opt_state = train_step(
            model,
            fcn_params["criterion_threshold"],
            optimizer,
            opt_state,
            x,
            logs,
        )
        losses.append(loss)
        if step % 1 == 0:
            print(f"Step {step} | Loss: {loss}")

    # Plot the losses
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Outer Step")
    plt.ylabel("Loss")
    plt.title("Outer Training Loss")
    plt.show()

    # Show how the mask evolves over time
    plt.figure()
    steps_to_plot = range(fcn_params["num_iterations"])
    for step in steps_to_plot:
        plt.plot(logs["masks"][step], label=f"Step {step}")
    plt.xlabel("Source Index")
    plt.ylabel("Probability")
    plt.title("Mask Evolution")
    plt.legend()
    plt.show()

    # Final mask:
    final_mask = logs["masks"][-1]
    best_sources = jnp.argsort(final_mask)[-10:]
    print("Best source indices:", best_sources)

if __name__ == "__main__":
    main()
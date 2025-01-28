# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: hybridOED
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test case D: FWI L1 norm as an criterion

# %%
import numpy as np
from jax import jit, vjp, vmap, pmap, random, value_and_grad, nn, profiler
import jax
from jax import numpy as jnp
import equinox as eqx
import optax
from matplotlib import pyplot as plt
from jaxdf.operators import compose
from jax.profiler import save_device_memory_profile



from jwave.geometry import Domain, Medium
from jwave.utils import display_complex_field

key = random.PRNGKey(42)  # Random seed


from hybridoed.forward import create_src_field, generate_2D_gridded_src_rec_positions
from hybridoed.oed import *


from jwave.geometry import Domain, Medium
from jwave.utils import display_complex_field

key = random.PRNGKey(42)  # Random seed


from hybridoed.forward import create_src_field
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
from jwave.signal_processing import apply_ramp, gaussian_window, smooth

from jwave.signal_processing import analytic_signal
from jaxdf.operators import gradient, functional


# %%
# save_device_memory_profile('memory_profile.json')

# profiler.start_trace('trace')

# %%
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
# Adjust sigma as needed; larger sigma = more blur
blurred_model = gaussian_filter(true_model, sigma=20)

# Visualize the blurred model
plt.figure()
plt.imshow(blurred_model, cmap='viridis')
plt.colorbar()
plt.title('Blurred Model')
plt.show()
print(blurred_model.shape)

# homogenous model of the velocity avegare

homogenous_model = jnp.ones(true_model.shape) * jnp.mean(true_model)

# plt.imshow(homogenous_model)

# %%
N = (70, 70)  # Grid size
dx = (1.0, 1.0)  # Spatial resolution
cfl=0.75

# Defining the domain
domain = Domain(N, dx)

source_freq = 200
source_mag = 1.3e-1
# source_mag = 1.3

medium = Medium(domain=domain, sound_speed=true_model, density=1000., pml_size=10)

# Time axis
time_axis = TimeAxis.from_medium(medium, cfl=cfl)
t = time_axis.to_array()
source_mag = source_mag / time_axis.dt

s1 = source_mag * jnp.sin(2 * jnp.pi * source_freq * t + 100)
signal = gaussian_window(apply_ramp(s1, time_axis.dt, source_freq), t, 0.5e-2, 1e-2)

# generate a ricker signal instead of a sinusoidal signal

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


# Parameters
# source_freq = 25.0  # Central frequency in Hz
# t = jnp.linspace(-0.1, 0.2, 1000)  # Time vector in seconds
t_shift = 0.005  # Time shift in seconds

# Generate the Ricker wavelet with time shift
signal = source_mag * ricker_wavelet(t, source_freq, t_shift)




src_coords_list, receiver_coords_list = generate_2D_gridded_src_rec_positions(N=(70, 70), num_sources=5, num_receivers=5)


num_sources = src_coords_list.shape[0]
sensors_positions = (receiver_coords_list[:,0],receiver_coords_list[:,1])
sensors = Sensors(positions=sensors_positions)
source_positions = (src_coords_list[:,0],src_coords_list[:,1])

print(sensors_positions)


# Show comprehensive simulation setup

fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 2]})

ax[0].imshow(medium.sound_speed.on_grid, cmap="gray")
ax[0].scatter(
    source_positions[1], source_positions[0], c="r", marker="x", label="sources"
)
ax[0].scatter(
    sensors_positions[1], sensors_positions[0], c="g", marker=".", label="sensors"
)
ax[0].legend(loc="lower right")
ax[0].set_title("Sound speed")
ax[0].axis("off")

ax[1].plot(signal, label="Source 1", c="k")
ax[1].set_title("Source signals")
#ax[1].get_yaxis().set_visible(False)

# %% [markdown]
# ## Setup for the test case C and FWI loss function

# %% [markdown]
#

# %%
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

# %%
# Load and format the Jacobians

# jacobians = jnp.load("curvel_jac_model_{}_data_5_5_5.npy".format(model_index))
# print(jacobians.shape)

# transposed_jacobians = jnp.transpose(jacobians, axes=(1, 2, 0, 3))

# print(transposed_jacobians.shape)
# stacked_array = jnp.stack([array.reshape(-1, array.shape[-1]) for array in transposed_jacobians])
# stacked_array.shape

# complex_stack_complete = []
# for src in stacked_array:
#     complex_stack = []
#     for row in src:
#         real_part = row.reshape(70,140)[:,:70]
#         imaginary_part = row.reshape(70,140)[:,70:]
#         complex_stack.append(real_part + 1j * imaginary_part)

#     # Convert to NumPy array if needed
#     complex_stack = jnp.array(complex_stack)

#     # print(complex_stack.shape)

#     # Flatten all rows into num_rows x 4900 matrix
#     num_rows = complex_stack.shape[0]
#     complex_reshaped = np.empty((num_rows, 4900), dtype=np.complex128)

#     for i, complex_matrix in enumerate(complex_stack):
#         complex_reshaped[i, :] = complex_matrix.flatten()
    
#     complex_stack_complete.append(complex_reshaped)

# complex_stack_complete = jnp.array(complex_stack_complete)
# print(complex_stack_complete.shape)

# complex_stack_complete_2D = jnp.vstack([array for array in complex_stack_complete])
# print(complex_stack_complete_2D.shape)



# %%
# Calculate the eigenvalue criterion for each Jacobian row to serves as input for the network

criterion_threshold = fcn_params["criterion_threshold"]

# C_sources_1e_3 = [] 
# for array in stacked_array:
#     C_sources_1e_3.append(eigenvalue_criterion(array, threshold=criterion_threshold))

# C_sources_1e_3 = jnp.array(C_sources_1e_3)

# del stacked_array

# %%
# FWI functions

src_signal = jnp.stack([signal])

@jit
def single_source_simulation(sound_speed, source_num):

        

    # if isinstance(source_num, int):
    x = lax.dynamic_slice(source_positions[0], (source_num,), (1,))
    y = lax.dynamic_slice(source_positions[1], (source_num,), (1,))
        # print("x, y","int", x,y)

    # else:
    #     x = [source_num[0].astype(jnp.int32)]
    #     y = [source_num[1].astype(jnp.int32)]
    #     print("x, y","array", x,y)
        
    sources = Sources((x, y), src_signal, dt=time_axis.dt, domain=domain)

    # Updating medium with the input speed of sound map
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=10)

    # Run simulations
    rf_signals = simulate_wave_propagation(
        medium, time_axis, sources=sources, sensors=sensors, checkpoint=True
    )
    return rf_signals[..., 0]

print(type(num_sources//2))
# print(len(source_trajectories[-1][0]))
p = single_source_simulation(medium.sound_speed, num_sources // 2)
# p = single_source_simulation(medium.sound_speed, source_trajectories[-1][0])
# 
# Visualize the acoustic traces
plt.figure(figsize=(6, 4.5))
maxval = jnp.amax(jnp.abs(p))
plt.imshow(
    p, cmap="RdBu_r", interpolation="nearest", aspect="auto"
)
plt.colorbar()
plt.title("Acoustic traces")
plt.xlabel("Sensor index")
plt.ylabel("Time")
plt.show()

mask = jnp.ones(domain.N)

mask = mask.at[sensors_positions[0], sensors_positions[1]].set(0.0)




def get_sound_speed(params):
    return params + compose(params)(nn.sigmoid) * mask



# %%
# # initial model is the blurred model
# params = blurred_model
# # params = homogenous_model
# # params = medium.sound_speed * 0.1 + 
# params

# probabilities = jnp.ones(num_sources)
# # probabilities = jnp.array([0. ,0. ,0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1.])

# def hilbert_transf(signal, noise=0.2):
#     x = jnp.abs(analytic_signal(signal))
#     return x


# loss_with_grad = value_and_grad(loss_func, argnums=0)

# batch_simulations = vmap(single_source_simulation, in_axes=(None, 0))
# p_data = batch_simulations(medium.sound_speed, jnp.arange(num_sources))
# print(f"Size of data [Source idx, Time, Sensor idx]: {p_data.shape}")

# plt.plot(p_data[0, :, 0])
# plt.plot(p_data[1, :, 1])
# plt.show()

# def smooth_fun(gradient):
#     # x = gradient.on_grid[..., 0]
#     x = gradient
#     for _ in range(1):
#         x = smooth(x, 2.0)
#     return x

# loss, gradient = loss_with_grad(params, source_num=2)
# gradient = smooth_fun(gradient)

# # Viualize
# plt.figure(figsize=(8, 6))
# plt.imshow(gradient, cmap="RdBu_r")
# plt.title("Smoothed gradient")
# plt.colorbar()
# plt.show()

# num_steps = 250

# # Define optimizer
# init_fun, update_fun, get_params = optimizers.adam(20.0, 0.9, 0.9)
# opt_state = init_fun(params)



# # Main loop
# pbar = tqdm(range(num_steps))
# _, key = random.split(key)
# batch_size = 10
# num_devices = 12

# reconstructions = []
# losshistory = []

# for k in pbar:
#     _, key = random.split(key)

#     lossval, opt_state = update(opt_state, key, k, probabilities)

#     # Perform update using multiple sources
#     # lossval, opt_state = update_multi(opt_state, key, k)

#     # Perform update using multiple sources in parallel
#     # lossval, opt_state = update_multi_pmap(opt_state, key, k)
    

#     ## For logging
#     new_params = get_params(opt_state)
#     reconstructions.append(get_sound_speed(new_params))
#     losshistory.append(lossval)
#     pbar.set_description("Loss: {}".format(lossval))


# true_sos = true_model
# vmin = np.amin(true_sos)
# vmax = np.amax(true_sos)
# # Visualize the final reconstruction
# reconstructions = jnp.array(reconstructions) 
# plt.figure(figsize=(8, 6))
# plt.imshow(reconstructions[-1], cmap="inferno", vmin=vmin, vmax=vmax)
# plt.colorbar()
# plt.title("Final reconstruction")
# plt.show()

# %%
# Fully Connected Neural Network
class FullyConnectedNN(eqx.Module):
    layers: list
    activations: list

    def __init__(self, input_size, hidden_size, num_hidden_layers, key):
        keys = jax.random.split(key, num_hidden_layers + 1)
        self.layers = [eqx.nn.Linear(input_size, hidden_size, key=keys[0])] + \
                      [eqx.nn.Linear(hidden_size, hidden_size, key=k) for k in keys[1:-1]] + \
                      [eqx.nn.Linear(hidden_size, input_size, key=keys[-1])]
        
        self.activations = [jax.nn.tanh] * num_hidden_layers + [jax.nn.sigmoid]

    def __call__(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x

# Differentiable Mask
def differentiable_mask(probabilities, sharpness=10.0):
    return jax.nn.sigmoid(sharpness * (probabilities - 0.5))

# def top_k_regularization(soft_mask, k=10):
#     top_k_values = jax.lax.top_k(soft_mask, k)[0]
#     penalty = jnp.sum(soft_mask) - jnp.sum(top_k_values)
#     return penalty

def fwi(probabilities):
    
    # _, key = random.split(key)
    # initial model is the blurred model
    params = blurred_model
    init_fun, update_fun, get_params = optimizers.adam(20.0, 0.9, 0.9)
    opt_state = init_fun(params)

    print(f"Size of data [Source idx, Time, Sensor idx]: {p_data.shape}")

    def loss_func(params, source_num):
        c0 = get_sound_speed(params)
        p = single_source_simulation(c0, source_num)
        data = p_data[source_num]
        # return jnp.mean(jnp.abs(hilbert_transf(p) -hilbert_transf(data)) ** 2)
        # L2 loss
        return jnp.mean((p - data) ** 2)

    loss_with_grad = value_and_grad(loss_func, argnums=0)

    # Define and compile the update function
    # @jit
    def update(opt_state, key, k, probabilities):
        v = get_params(opt_state)
        src_num = random.choice(key, num_sources)
        lossval, gradient = loss_with_grad(v, src_num)
        # gradient = smooth_fun(gradient)
        gradient *= probabilities[src_num]
        return lossval, update_fun(k, gradient, opt_state)

    for k in range(55):

        lossval, opt_state = update(opt_state, key, k, probabilities)

        ## For logging
        new_params = get_params(opt_state)
    
    return new_params, lossval



# Differentiable Loss Function
def differentiable_loss_fn(model, x, criterion_threshold ,sharpness=10.0, mask_penalty=0.1):
    probabilities = model(x)  # Predict probabilities
    soft_mask = differentiable_mask(probabilities, sharpness)  # Generate soft mask
    
    

    # multiplier = matrix.shape[0] // soft_mask.shape[0]
    # soft_mask = jnp.repeat(soft_mask, multiplier, axis=0)

    # weighted_matrix = soft_mask[:, None] * matrix  # Apply mask to matrix rows

    regularization_loss = mask_penalty * jnp.mean(soft_mask * (1 - soft_mask))  # Encourage binary mask
    norm_loss = jnp.linalg.norm(soft_mask, ord=1)
    # top_k_loss = top_k_regularization(soft_mask, k=fcn_params["number_of_k"])

    # singular_loss = eigenvalue_criterion(weighted_matrix, threshold=criterion_threshold)
    inverted_model, lossval = fwi(probabilities)
    
    fwi_loss = jnp.linalg.norm(inverted_model - true_model, ord=1)

    # fwi_loss = lossval



    # return -(singular_loss) + params["regularisation_loss"]*regularization_loss + params["norm_loss"]*norm_loss + params["top_k_loss"]*top_k_loss
    return fwi_loss + fcn_params["regularisation_loss"]*regularization_loss + fcn_params["norm_loss"]*norm_loss 



# %%
# Example Usage
key = jax.random.PRNGKey(42)
input_size = fcn_params["num_sources"]
hidden_size = fcn_params["hidden_size"]
num_hidden_layers = fcn_params["num_hidden_layers"]
model = FullyConnectedNN(input_size, hidden_size, num_hidden_layers, key)

print("Model:", model)

# matrix = jax.random.normal(key, (100, 50))  # Matrix with 100 rows and 50 columns
# matrix = complex_stack_complete_2D
# x = jax.random.normal(key, (input_size,))  # Input to the network
# x = C_sources_1e_3
x = jnp.ones(input_size)
# x = (x - jnp.mean(x)) / jnp.std(x)
# x = (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))

# %%
def train_step(model, criterion_threshold, optimizer, opt_state, x, logs):
    # Compute loss and gradients
    loss_and_grad_fn = eqx.filter_value_and_grad(differentiable_loss_fn)
    loss, grads = loss_and_grad_fn(model, x, criterion_threshold=criterion_threshold, sharpness=fcn_params["differentiable_mask_sharpness"], mask_penalty=1.0)
    
    if fcn_params["print_gradients"]:  
        jax.tree_util.tree_map(lambda g: print("Gradient shape:", g.shape, "Gradient values:", g), grads)
        jax.tree_util.tree_map(lambda g: print(f"Gradient shape: {g.shape}, Min: {jnp.min(g)}, Max: {jnp.max(g)}, Mean: {jnp.mean(g)}"), grads)

    # Update optimizer state and model
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)

    # Log the network output (predicted probabilities)
    probabilities = model(x)  # Network output
    logs["probabilities"].append(probabilities)

    # Log the mask M
    soft_mask = differentiable_mask(probabilities, sharpness=fcn_params["differentiable_mask_sharpness"])  # Differentiable mask
    logs["masks"].append(soft_mask)

    return loss, model, opt_state


# %%

# Initialize the model and optimizer
key = jax.random.PRNGKey(42)
model = FullyConnectedNN(input_size, hidden_size, num_hidden_layers, key)
optimizer = optax.adamw(fcn_params["learning_rate"])  # Adam optimizer
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))  # Initialize optimizer state
initial_predictions = model(x)  # Initial predictions

batch_simulations = vmap(single_source_simulation, in_axes=(None, 0))
p_data = batch_simulations(medium.sound_speed, jnp.arange(num_sources))

# Training loop
num_steps = fcn_params["num_iterations"]
losses = []
logs = {"probabilities": [], "masks": []}
for step in range(num_steps):
    loss, model, opt_state = train_step(model, criterion_threshold, optimizer, opt_state, x, logs)
    losses.append(loss)

    if step % 1 == 0:
        print(f"Step {step}, Loss: {loss}")

# Plot the losses
plt.figure()
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()

# %%
# Print the first few steps for inspection
for i, (probs, mask) in enumerate(zip(logs["probabilities"], logs["masks"])):
    print(f"Step {i}: Probabilities: {probs}")
    print(f"Step {i}: Mask: {mask}")
    if i > 5:  # Limit to first few steps for readability
        break

# Example: Plot the first mask at different training steps
steps_to_plot = [i for i in range(num_steps)]  # Select steps to visualize
plt.figure()

for step in steps_to_plot:
    plt.title(f"Output mask at different steps")
    plt.plot(logs["masks"][step], label=f"Step {step}")
    # plt.colorbar(label="Mask Values")
    plt.xlabel("Mask index")
    plt.ylabel("Probability")

plt.plot(x, ".-", label="Original input")
plt.plot(initial_predictions,"*-", label="Initial predictions")
# put the legend outside of the graph, to the left
plt.legend(loc='center left', bbox_to_anchor=(-0.5, 0.5))

# add the params dictionary to the plot, on the right, one key per line
plt.text(1.05, 0.5, '\n'.join([f"{key}: {value}" for key, value in fcn_params.items()]), transform=plt.gca().transAxes)

plt.show()

# %%
final_mask = logs["masks"][-1]
best_sources_C = jnp.argsort(final_mask)[-10:]
print("best source index", best_sources_C)

# %%
# model = fwi(final_mask)

# %%

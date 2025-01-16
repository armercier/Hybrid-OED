# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: hybridOED
#     language: python
#     name: python3
# ---

# # Test case B: Continuous optimization of the source and receiver position

# +
import numpy as np
from jax import jit, vjp
import jax
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt
import optax
import equinox as eqx


from jwave.geometry import Domain, Medium
from jwave.utils import display_complex_field

key = random.PRNGKey(42)  # Random seed


from hybridoed.forward import create_src_field
from hybridoed.oed import *


# +
# N = (32, 32)  # Grid size
N = (70, 70)
# dx = (0.1, 0.1)  # Spatial resolution
dx = (1.0, 1.0)  # Spatial resolution
f = 400
omega = 2*jnp.pi*f  # Wavefield omega = 2*pi*f

# Defining the domain
domain = Domain(N, dx)

# +
x = N[0]//2 + 0.01
y = N[1]//2 + 0.01
src_coord = jnp.array([[y, x]]).astype(jnp.float32)
# receiver_coords = jnp.array([[8.1, 28.1]]).astype(jnp.float32)
# receiver_coords = jnp.array([[8.1, 2*28.1]]).astype(jnp.float32)

def generate_2D_gridded_src_rec_positions_test_case_B(N=(70, 70), num_sources=20, num_receivers=10):
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
    recv_x = jnp.linspace(5 + (Nx - 10) / (2 * num_receivers), Nx - 5 - (Nx - 10) / (2 * num_receivers), num_receivers, dtype=jnp.float32)
    recv_y = jnp.linspace(5 + (Ny - 10) / (2 * num_receivers), Ny - 5 - (Ny - 10) / (2 * num_receivers), num_receivers, dtype=jnp.float32)

    # Create 2D grid coordinates for sources and receivers
    src_coords = jnp.array([[x, y] for x in src_x for y in src_y], dtype=jnp.int32)
    recv_coords = jnp.array([[x, y] for x in recv_x for y in recv_y], dtype=jnp.int32)

    return src_coords, recv_coords


src_coords_list, receiver_coords_list = generate_2D_gridded_src_rec_positions_test_case_B(N=(70, 70), num_sources=3, num_receivers=5)
src_coords_list = src_coords_list 
receiver_coords_list = receiver_coords_list 


num_sources = src_coords_list.shape[0]
# sensors_positions = (receiver_coords_list[:,0],receiver_coords_list[:,1])
# sensors = Sensors(positions=sensors_positions)
# source_positions = (src_coords_list[:,0],src_coords_list[:,1])

# print(sensors_positions)

src_coord = src_coords_list + 0.1
print(src_coord)
print(src_coord[0][0], src_coord[0][1])

# many receivers
# receiver_coords = jnp.array([[8.1, 0.125*N[1]+0.1]]).astype(jnp.float32)
receiver_coords = receiver_coords_list + 0.1
print(receiver_coords)

# receiver_coords = jnp.array([[8.1, 0.125*N[1]+0.1],[8.1, 0.25*N[1]+0.1], [8.1, 0.5*N[1]+0.1], [8.1, 0.75*N[1]+0.1], [8.1, 0.90*N[1]+0.1],]).astype(jnp.float32)

src = create_src_field(N, src_coord[0][0], src_coord[0][1], domain, omega)

# Plotting
_ = display_complex_field(src)

# +
# load data1.npy
model = np.load('model1.npy')
print(model.shape)

plt.imshow(model[7,0,:,:])
plt.colorbar()
plt.show()

# downsample to 32 by 32
# model = model[:,:,6::2,6::2]
print(model.shape)

plt.imshow(model[7,0,:,:])
plt.colorbar()

# +
# Constructing medium physical properties
sound_speed = jnp.zeros(N)

density = 1000.  # sound_speed*0 + 1.
attenuation = 0.0  # density*0

# sound_speed = jnp.full(N, 1540.)
sound_speed = jnp.array(model[7,0,:,:])

pml_size = 15

# make a square in the middle of the domain
# sound_speed = sound_speed.at[N[0]//3:N[0]//3+8, N[1]//3:int(N[1]//4 * 2.5)].set(1000.)



true_medium = Medium(domain=domain, sound_speed=sound_speed,pml_size=pml_size)

# +
src_coord = src_coord.at[4].set([20.1,20.1])

# New row to add
new_row = jnp.array([50.1, 50.1], dtype=jnp.float32)

# Add the new row at index 5
src_coord = jnp.insert(src_coord, 6, new_row, axis=0)

print(src_coord)


# +
# plot the medium
plt.figure()
plt.imshow(true_medium.sound_speed.on_grid)
# add the receiver position
# plot all the receivers

plt.scatter(*receiver_coords.T[::-1], color='r', marker='x')

# plt.scatter(*receiver_coords[0][::-1], color='r', marker='x')
plt.scatter(*src_coord[::-1].T, color='b', marker='*')
plt.colorbar()
plt.title("Sound speed")
plt.legend(["Receiver", "Source"], loc='upper left')
plt.show()

# +
from jwave.acoustics.time_harmonic import helmholtz, helmholtz_solver

@jit
def solve_helmholtz_general(sound_speed, src_coord, omega):
    src = create_src_field(N, src_coord[0], src_coord[1], domain, omega)
    medium = Medium(domain=domain, sound_speed=sound_speed, density=1000., pml_size=pml_size)
    field = helmholtz_solver(medium, omega, src)
    return field.on_grid.squeeze()


# -

true_field = solve_helmholtz_general(sound_speed, [y,x], omega)

_ = display_complex_field(true_field)


# +
def extract_receiver_values(field, receiver_coords):
    receiver_coords = receiver_coords.astype(jnp.int32)
    return jnp.array([field[coord[0], coord[1]] for coord in receiver_coords])

def extract_receiver_values_interpolated(field, receiver_coords):
    # Assume receiver_coords is a float array
    x_floor = jnp.floor(receiver_coords[:, 0]).astype(jnp.int32)
    y_floor = jnp.floor(receiver_coords[:, 1]).astype(jnp.int32)
    x_ceil = jnp.ceil(receiver_coords[:, 0]).astype(jnp.int32)
    y_ceil = jnp.ceil(receiver_coords[:, 1]).astype(jnp.int32)

    # Interpolation weights
    wx = receiver_coords[:, 0] - x_floor
    wy = receiver_coords[:, 1] - y_floor

    # Bilinear interpolation
    top_left = field[x_floor, y_floor]
    top_right = field[x_floor, y_ceil]
    bottom_left = field[x_ceil, y_floor]
    bottom_right = field[x_ceil, y_ceil]

    interpolated_values = (
        (1 - wx) * (1 - wy) * top_left +
        wx * (1 - wy) * top_right +
        (1 - wx) * wy * bottom_left +
        wx * wy * bottom_right
    )

    # Return real and imaginary parts separately
    # print(jnp.real(interpolated_values).shape)
    return jnp.real(interpolated_values), jnp.imag(interpolated_values)

def compute_jacobian_at_receivers_vjp(sound_speed, src_coords, omega, receiver_coords):
    def field_at_receivers(sound_speed, src_coord, receiver_coords, omega):
        field = solve_helmholtz_general(sound_speed, src_coord, omega)
        rec_real, rec_imag = extract_receiver_values_interpolated(field, receiver_coords)
        return rec_real, rec_imag
    
    jacobians_real = []
    jacobians_imag = []
    
    for src_coord in src_coords:
        # Compute VJP for the given source
        (y_real, y_imag), vjp_fn = vjp(lambda s: field_at_receivers(s, src_coord,receiver_coords, omega), sound_speed)

        print("y_real",y_real.shape)
        
        # Calculate VJP for both real and imaginary parts
        # jacobian_real = vjp_fn((jnp.ones_like(y_real), jnp.zeros_like(y_imag)))[0]
        # jacobian_imag = vjp_fn((jnp.zeros_like(y_real), jnp.ones_like(y_imag)))[0]

        # Compute the full Jacobian for each receiver
        jacobian_real = []
        jacobian_imag = []

        for i in range(y_real.shape[0]):  # Iterate over each receiver
            cotangent_real = jnp.zeros_like(y_real).at[i].set(1.0)  # One-hot for real part
            cotangent_imag = jnp.zeros_like(y_imag).at[i].set(1.0)  # One-hot for imaginary part
            
            jac_real = vjp_fn((cotangent_real, jnp.zeros_like(cotangent_imag)))[0]
            jac_imag = vjp_fn((jnp.zeros_like(cotangent_real), cotangent_imag))[0]

            jacobian_real.append(jac_real)
            jacobian_imag.append(jac_imag)

        jacobians_real.append(jnp.stack(jacobian_real, axis=0))  # Stack (25, 70, 70)
        jacobians_imag.append(jnp.stack(jacobian_imag, axis=0))  # Stack (25, 70, 70)

        # print("jacobian_real",jacobian_real.shape)
        print("jacobian_real tuple length",len(jacobian_real))
        print("jacobian_real shape",jacobian_real[0].shape)
        print("jacobian_real shape",jacobian_real[1].shape)
        
        # jacobians_real.append(jacobian_real)
        # jacobians_imag.append(jacobian_imag)
    
    # Stack the Jacobians for each source
    return jnp.array(jacobians_real), jnp.array(jacobians_imag)

def determiant_of_approximated_hessian(Jacobian):
    """
    Compute the determinant of the approximated Hessian matrix

    Parameters:
    - Jacobian (np.ndarray): The Jacobian matrix

    Returns:
    - determinant (float): The determinant of the approximated Hessian matrix
    """
    
    # copmut the sum of all the columns
    # print("jacobian shape",Jacobian.shape)
    # return jnp.abs(jnp.sum(jnp.abs(jnp.sum(Jacobian, axis=0))))
    # return jnp.sum(jnp.abs(jnp.diag(Jacobian.T @ Jacobian)))


    # return jnp.log10(jnp.linalg.cond(Jacobian.T @ Jacobian))

    # eig_vals, eig_vec = jnp.linalg.eigh(jnp.conj(Jacobian).T @ Jacobian)

    # count the number of eigenvalues that are greater than 1e-10
    # threshold = 1e-14
    # return jnp.sum(eig_vals > threshold)

    
    # return jnp.sum(jax.nn.sigmoid(8e12 * (eig_vals - threshold)))



    # return jnp.sum(jax.scipy.special.entr(jnp.real(eig_vals) / jnp.sum(jnp.real(eig_vals))))

    # return jnp.log10(jnp.var(eig_vals)) + 1e-5

    # hessian =   Jacobian.T @ Jacobian
    # regularized_hessian = hessian + 0.98 * jnp.eye(hessian.shape[0])
    # return ((jnp.linalg.det(hessian)))
    # return jnp.abs(scaled_det(hessian))
    # get the largest eigenvalue
    # lambda_max, _ = power_method(hessian)
    # return -jnp.real(jnp.trace((hessian/lambda_max)))

    return -eigenvalue_criterion(Jacobian, threshold=1e-2)

def compute_penalty(src_coord, domain_size, penalty_weight=1.0, epsilon=1e-1):
    """
    Computes a symmetric penalty based on the distance of the source to each domain border.
    
    Parameters:
    - src_coord: The coordinates of the source (x, y).
    - domain_size: The size of the domain (N_x, N_y).
    - penalty_weight: The weight of the penalty term.
    - epsilon: A small constant to avoid division by zero.
    
    Returns:
    - penalty: The symmetric penalty value considering all borders equally.
    """
    N_x, N_y = domain_size
    x, y = src_coord[0]  # Correct access to source coordinates

    # Calculate distance to each border
    d_left = x
    d_right = N_x - x
    d_top = y
    d_bottom = N_y - y

    # Ensure distances are non-zero to avoid division by zero
    d_left = jnp.amax(jnp.array([d_left, epsilon]))
    d_right = jnp.amax(jnp.array([d_right, epsilon]))
    d_top = jnp.amax(jnp.array([d_top, epsilon]))
    d_bottom = jnp.amax(jnp.array([d_bottom, epsilon]))

    # d_right = jnp.max(d_right, epsilon)
    # d_top = jnp.max(d_top, epsilon)
    # d_bottom = jnp.max(d_bottom, epsilon)

    # Compute penalties for each border
    penalty_left = penalty_weight / d_left
    penalty_right = penalty_weight / d_right
    penalty_top = penalty_weight / d_top
    penalty_bottom = penalty_weight / d_bottom

    # Combine the penalties symmetrically, ensuring equal weight to each border
    total_penalty = (penalty_left + penalty_right + penalty_top + penalty_bottom) / 4.0

    return total_penalty

def objective_function(args): 

    src_coord, receiver_coords = args  
    # src_coord = args

    # delta_new_src_list = args(jnp.array(src_coord).ravel()).reshape(-1, 2)
    # print(new_src_list)
    # make sure that every value in new_src_list is a positive integer
    # delta_new_src_list = jnp.abs(delta_new_src_list)
    # new_src_list = jnp.array(src_coord) + delta_new_src_list 
    # new_src_list = new_src_list.round()
    # new_src_list = new_src_list.astype(int)

    # new_src_list = new_src_list.tolist()
    # print(new_src_list)

    # transform the args into all integers
    # print('raw arg',args)
    # args = jnp.round(args).astype(jnp.int32)
    # print('rounded args',args)

    J = compute_jacobian_at_receivers_vjp(sound_speed, src_coord, omega, receiver_coords)
    print("before stacking",J[0].shape)



    # plt.figure()
    # plt.imshow(jnp.real(J[0]))
    # plt.colorbar()
    # plt.title("Jacobian")
    # plt.show()

    # J = compute_jacobian(new_src_list, sound_speed, omega)


    # J_transformed, c_list = haar_transform_pywavelet(J, N, level=None)
    # diag_hessian = jnp.real(np.sum(np.conj(J_transformed) * J_transformed, axis=0))
    # print(diag_hessian.shape)
    # mask = create_mask(diag_hessian, 0.9)
    # # number_of_coeef_kept = jnp.sum(mask)
    # # print(mask)
    # print("number of true element in mask", jnp.sum(mask))
    # oed_criterion_result = jnp.sum(mask)

    # oed_criterion_result = jnp.abs(dummy_OED_criterion(J[0]))

    penality = compute_penalty(src_coord, N, penalty_weight=1.0, epsilon=2.6)

    # reshaping for average jacobian over all receviers
    # ravel every element and stack 
    # if J[0].shape[0] == 1:
    #     J = jnp.vstack([jnp.ravel(j) for j in J])
    #     J = jnp.array(J) 
    #     if len(J) > 1:
    #         J = jnp.vstack(J)
    #         # print(J.shape)
    #     print("here")

    # else:
    #     j_temp = []
    #     for r_c in J:
    #         for j in r_c:
    #             # print(j.shape)
    #             j_temp.append(jnp.ravel(j))

    #     J = jnp.vstack(j_temp)
        
    # full jacobian
    Ja_real = jnp.array(J[0])
    print(Ja_real.shape)
    Ja_real = Ja_real.reshape(-1, Ja_real.shape[-2]*Ja_real.shape[-1])
    print(Ja_real.shape)

    Ja_imag = jnp.array(J[1])
    print(Ja_imag.shape)
    Ja_imag = Ja_imag.reshape(-1, Ja_imag.shape[-2]*Ja_imag.shape[-1])
    print(Ja_imag.shape)



    J_complex = Ja_real + 1j*Ja_imag


    # eig_vals, eig_vec = jnp.linalg.eigh(jnp.conj(J).T @ J)

    # plt.figure()
    # plt.semilogy(jnp.real(eig_vals).sort()[::-1])
    # plt.title("eigen values")
    # plt.show()

    # print("after stacking",J.shape)


    oed_criterion_result = determiant_of_approximated_hessian(J_complex)

    penality = 0
    return oed_criterion_result + 0*penality 
    # return penality

# +
# src_coord, receiver_coords = fargs  

# J = compute_jacobian_at_receivers_vjp(sound_speed, src_coord, omega, receiver_coords)



# +

# reshape J[0] from (4,25,70,70) by (100,4900)
# Ja_real = jnp.array(J[0])
# print(Ja_real.shape)
# Ja_real = Ja_real.reshape(-1, Ja_real.shape[-2]*Ja_real.shape[-1])
# print(Ja_real.shape)

# Ja_imag = jnp.array(J[1])
# print(Ja_imag.shape)
# Ja_imag = Ja_imag.reshape(-1, Ja_imag.shape[-2]*Ja_imag.shape[-1])
# print(Ja_imag.shape)



# J_complex = Ja_real + 1j*Ja_imag

# print(J_complex.shape, type(J_complex)) 

# -

# for i in range(9):
#     plt.figure()
#     plt.imshow(jnp.real(J_complex)[i].reshape(70,70))
#     plt.colorbar()
#     plt.title("Jacobian")
#     plt.show()


# ### Calculate the loss landscape

# +
# calculate the loss landscape for the position of the source
# from tqdm import tqdm

# pbar = tqdm(total=N[0] * N[1])
# loss_landscape = jnp.zeros(N)

# for i in range(N[0]):
#     for j in range(N[1]):
#         source_coord_temp = jnp.array([[i+0.1, j+0.1]]).astype(jnp.float32)
#         obj = objective_function((source_coord_temp, receiver_coords))
#         # print("obj",obj)
#         loss_landscape = loss_landscape.at[i, j].set(obj)
#         # print("progression", i*N[1] + j, "out of", N[0]*N[1], loss_landscape[i, j])
#         pbar.set_description(f"Loss: {loss_landscape[i, j]:.4f}")
#         pbar.update(1)        
# pbar.close()

# +
# plt.imshow(loss_landscape)
# plt.colorbar()
# plt.title("Loss landscape of the position of the source")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
# print(jnp.amax(loss_landscape))
# -

# ## Optimization loop

fargs = (src_coord, receiver_coords)
# fargs = [src_coord]
print(fargs)
print(fargs[0])
print(fargs[1])
print(fargs[0].shape)
print(fargs[1].shape)
print(fargs[0][0].shape)
print(fargs[1][1].shape)


def loss_fn(fargs):
    objective = objective_function(fargs)
    return objective


# +
# loss_fn(fargs)
# -

def grad(fargs):
    "Computes gradient of loss function wrt fargs"
    partial = lambda fargs: loss_fn(fargs)
    return eqx.filter_value_and_grad(partial)(fargs)


def step(fargs, opt_state, opt_update):
    lossval, grads = grad(fargs)

    # Print gradients for debugging
    def print_grads(grads):
        print("Gradients in step function:")

        def print_leaf(x):
            if x is None:
                print("None")
            else:
                print(x)
        jax.tree_util.tree_map(print_leaf, grads)
    
    print_grads(grads)
    # grads = (-grads[0], -grads[1])
    updates, opt_state = opt_update(grads, opt_state)
    fargs = eqx.apply_updates(fargs, updates)
    
    # Define the domain boundaries (adjust as necessary)
    x_min, x_max = 0.1, sound_speed.shape[1] - 1.1
    y_min, y_max = 0.1, sound_speed.shape[0] - 1.1

    # Clip the positions of the sources and receivers with respect to the pml_size
    x_min, x_max = pml_size * dx[0]*0.1 + 1.1, sound_speed.shape[1] - (pml_size * dx[0]*0.1 + 1.1)
    y_min, y_max = pml_size * dx[1]*0.1 + 1.1, sound_speed.shape[0] - (pml_size * dx[1]*0.1 + 1.1)
    
    # Clip the positions of the sources and receivers
    # fargs = (
    #     jnp.array([[jax.tree_util.tree_map(lambda x: jnp.clip(x, y_min, y_max), fargs[0][0][0]),
    #     jax.tree_util.tree_map(lambda x: jnp.clip(x, x_min, x_max), fargs[0][0][1])]]),
    #     jnp.array([[jax.tree_util.tree_map(lambda x: jnp.clip(x, y_min, y_max), fargs[1][1][0]),
    #     jax.tree_util.tree_map(lambda x: jnp.clip(x, x_min, x_max), fargs[1][1][1])]]) 
    # )

    def clip_source(coords, x_min, x_max, y_min, y_max):
        x_clipped = jnp.clip(coords[0], x_min, x_max)
        y_clipped = jnp.clip(coords[1], y_min, y_max)
        return jnp.array([x_clipped, y_clipped])

    # Vectorize the clipping function along the sources dimension
    vmap_clip_source = jax.vmap(clip_source, in_axes=(0, None, None, None, None))

    # Apply the clipping to fargs[0] and ensure the shape is preserved
    fargs = (
        vmap_clip_source(fargs[0], x_min, x_max, y_min, y_max),  # Clip fargs[0]
        fargs[1]  # Leave fargs[1] unchanged or apply similar processing if needed
    )


    # replace the receriver coords by fix value
    # fargs = (fargs[0], receiver_coords)
    print("fargs inside",fargs[0].shape)
    
    return lossval, fargs, opt_state, grads


# +
optimiser = optax.adam(learning_rate=0.9)
opt_state = optimiser.init(eqx.filter(fargs, eqx.is_array))
opt_update = optimiser.update
n_steps = 20

print("Initial source and receiver coordinates:", fargs)
# print("Initial optimizer state:", opt_state)

# +
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time  # Optional, to simulate time between iterations

# Create the figure and two subplots outside the loop
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

lossvals = []
source_trajectories = []
receiver_trajectories = []

for i in range(n_steps):
    tic = time.time()

    lossval, fargs, opt_state, grads = step(fargs, opt_state, opt_update)
    lossvals.append(lossval)
    tac = time.time()
    print(f"Step {i}, Loss: {lossval}, Time: {tac - tic}")

    # Record the current positions of sources and receivers
    # source_trajectories.append((fargs[0][0][0], fargs[0][0][1]))  # Source
    # receiver_trajectories.append((fargs[1][1][0], fargs[1][1][1]))  # Receiver

    source_trajectories.append(fargs[0])  # Source
    receiver_trajectories.append(receiver_coords[0])  # Receiver



    # receiver_trajectories.append(receiver_coords[0],receiver_coords[1])  # Receiver

    # Clear the current axes, keeping the figure intact
    ax1.cla()
    ax2.cla()

    if i % 1 == 0:
        print(f"Step {i}, Loss: {lossval}")

        # Plot the sound speed on the first subplot
        im = ax1.imshow(sound_speed)
        ax1.set_title("Updated source and receiver coordinates")

        # Plot the trajectories of the sources and receivers
        source_positions = list(zip(*source_trajectories))
        receiver_positions = list(zip(*receiver_trajectories))

        print("source_trajectories",source_trajectories)
        print("source_positions",source_positions)
        
        # for positions in source_positions:
        #     ax1.plot(positions[0][1], positions[0][0], 'b-', label='Source Trajectory')

        for traj in source_trajectories:
            # for src in traj:
                # ax1.plot(src[1], src[0], 'b-', label='Source Trajectory')
                # print(src[1], src[0])
            ax1.scatter(*traj.T[::-1], color='b', marker='.')

        # ax1.plot(source_positions[1], source_positions[0], 'b-', label='Source Trajectory')
        ax1.plot(receiver_positions[1], receiver_positions[0], 'r-', label='Receiver Trajectory')

        # Plot the current positions of the source and receiver
        # ax1.scatter(fargs[0][0][1], fargs[0][0][0], marker='*', color='blue', label='Current Source Position')
        # plot all the src positions in fargs
        # print(fargs[0][0])
        
        # print("000",fargs[0][0][0])
        
        for src in fargs[0]:
            ax1.scatter(*src.T[::-1], color='b', marker='*')
    
    
        # ax1.scatter(fargs[1][1][1], fargs[1][1][0], marker='x', color='red', label='Current Receiver Position')
        ax1.scatter(*receiver_coords.T[::-1], color='r', marker='x')
        # ax1.scatter(receiver_coords[1], receiver_coords[0], marker='o', color='blue', label='Current Receiver Position')

        print(fargs)

        # ax1.legend()

        # Plot the loss values over time on the second subplot
        ax2.plot(lossvals, label='Loss')
        ax2.set_title("Loss over iterations")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")
        ax2.legend()
    
        # Update the plot display
        # clear_output(wait=True)
        # display(fig)
        # plt.savefig(f"opti_movie/step_{i}.png")

        # Force a draw before saving and displaying the figure
        fig.canvas.draw()

        # Save the figure after drawing everything
        # plt.savefig(f"opti_movie/step_{i}.png")

        # Clear the output, display the updated figure
        # clear_output(wait=True)
        # display(fig)

        # Pause to allow plot to refresh
        time.sleep(0.1)


        # plt.pause(0.01)  # Slight pause to allow plot to update
        # time.sleep(0.1)  # Optional, simulates time between iterations

# Show the final plot
plt.show()

print("Final Loss:", lossvals[-1])

# save the source_trajectories, receiver_trajectories and lossvals

jnp.save('source_trajectories.npy', source_trajectories)
jnp.save('receiver_trajectories.npy', receiver_trajectories)
jnp.save('lossvals.npy', lossvals)


# +
# Read the source_trajectories, receiver_trajectories and lossvals from .np files and plot them as above

# source_trajectories = jnp.load('source_trajectories.npy')
# receiver_trajectories = jnp.load('receiver_trajectories.npy')
# lossvals = jnp.load('lossvals.npy')

# # Create the figure and two subplots outside the loop
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # Clear the current axes, keeping the figure intact
# ax1.cla()
# ax2.cla()

# # Plot the sound speed on the first subplot
# im = ax1.imshow(sound_speed)
# ax1.set_title("Updated source and receiver coordinates")

# # Plot the trajectories of the sources and receivers
# source_positions = list(zip(*source_trajectories))
# receiver_positions = list(zip(*receiver_trajectories))

# for traj in source_trajectories:
#     ax1.scatter(*traj.T[::-1], color='b', marker='.')

# # ax1.plot(receiver_positions[1], receiver_positions[0], 'r-', label='Receiver Trajectory')

# # Plot the current positions of the source and receiver
# for src in source_trajectories[-1]:
#     ax1.scatter(*src.T[::-1], color='b', marker='*')

# ax1.scatter(*receiver_coords.T[::-1], color='r', marker='x')

# # Plot the loss values over time on the second subplot
# ax2.plot(lossvals, label='Loss')
# ax2.set_title("Loss over iterations")
# ax2.set_xlabel("Iteration")
# ax2.set_ylabel("Loss")
# ax2.legend()

# # Update the plot display
# plt.show()

# print("Final Loss:", lossvals[-1])





# +
# from PIL import Image
# import os

# # Path to the folder containing the images
# image_folder = 'opti_movie'
# output_gif = 'opti_movie_normalize_trace.gif'

# # Get all images from the folder, assuming they are named as 'step_#.png'
# images = [img for img in os.listdir(image_folder) if img.startswith("step_") and img.endswith(".png")]
# images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by the number after 'step_'

# # Load all images into a list
# frames = [Image.open(os.path.join(image_folder, img)) for img in images]

# # Save as an animated GIF
# frames[0].save(output_gif, format='GIF',
#                append_images=frames[1:],
#                save_all=True,
#                duration=250,  # 250ms between frames (adjust FPS here)
#                loop=0)  # loop=0 means infinite loop

# print(f"GIF saved as {output_gif}")
# -



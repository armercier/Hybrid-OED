from jax import config as jax_config
jax_config.update("jax_disable_jit", True)
jax_config.update("jax_enable_x64", False)  # match your local setup
import jax.numpy as jnp

def test_sequential_oed():
    from hybridoed.oed import iterative_selection_no_reselection

    
    loaded_data = jnp.load('tests/integration/arrays_complete.npz')


    # # # Access arrays by index
    loaded_arrays = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data))]

    

    # vstack teh arrays
    J_c = (jnp.vstack(loaded_arrays))
    print("Checksum:", jnp.array(J_c).sum())
    if J_c.shape != (1000, 32768):
        raise ValueError(f"Expected 1000 rows, got {J_c.shape[0]} and 32768 columns, got {J_c.shape[1]}")
    
    print(f"J_c shape: {J_c.shape}")
    print(f"J_c min: {jnp.min(J_c)}, max: {jnp.max(J_c)}, mean: {jnp.mean(J_c)}")

    # Run the algorithm with logging
    num_rows = 9
    threshold = 10e-22
    J_o_final, O_final, mask_history, criterion_log = iterative_selection_no_reselection(J_c, num_rows,n_freq=5,n_receivers=20, threshold=threshold, sharpness="MODJO")
    

    print("O_final shape: ", O_final.shape)

    # Extract the selected row indices
    selected_rows = jnp.where(O_final.flatten() == 1)[0]

    print(selected_rows)

    ordered_list = []
    for iteration, row_idx in enumerate(selected_rows):
        ordered_list.append(row_idx.item())

    assert ordered_list[0] == [0, 3, 6, 108, 176, 318, 358, 421, 550][0]


# # Run the test
# test_sequential_oed()
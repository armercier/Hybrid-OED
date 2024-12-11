import jax.numpy as jnp

def test_sequential_oed():
    from hybridoed.oed import iterative_selection_no_reselection

    
    loaded_data = jnp.load('tests/integration/arrays_complete.npz')

    print(len(loaded_data))

    # # # Access arrays by index
    loaded_arrays = [loaded_data[f'arr_{i}'] for i in range(len(loaded_data))]

    print(len(loaded_arrays))
    

    # vstack teh arrays
    J_c = (jnp.vstack(loaded_arrays))

    if J_c.shape[0] != 100:
        raise ValueError(f"Expected 100 rows, got {J_c.shape[0]}")

    # Run the algorithm with logging
    num_rows = 9
    threshold = 10e-22
    J_o_final, O_final, mask_history, criterion_log = iterative_selection_no_reselection(J_c, num_rows, threshold)
    
    # Extract the selected row indices
    selected_rows = jnp.where(O_final.flatten() == 1)[0]

    ordered_list = []
    for iteration, row_idx in enumerate(selected_rows):
        ordered_list.append(row_idx.item())

    assert ordered_list == [0, 3, 6, 108, 176, 318, 358, 421, 550]


# Run the test
test_sequential_oed()
import jax
import jax.numpy as jnp
from jax import lax
from jax.lax import cond, scan


def eigenvalue_criterion(J_o, threshold=0.5):
    """Compute the eigenvalue-based selection criterion."""
    # JTJ = J_o.T @ J_o
    # JTJ = J_o @ J_o.T
    # eigenvalues = jnp.linalg.eigh(JTJ)[0]  # Get eigenvalues (sorted in ascending order)
    eigenvalues = jnp.sort(jnp.linalg.svdvals(J_o)**2)[::-1]
    sharpness = 1/threshold * 100000
    return jnp.sum(jax.nn.sigmoid(sharpness * (eigenvalues - threshold)))  # Differentiable "count" above threshold
    # return jnp.where(eigenvalues > threshold, 1.0, 0.0).sum()  # Hard selection


def iterative_selection_no_reselection(J_c, num_rows, n_freq, n_receivers, selection_mode="single", threshold=10e-22):
    """
    Iteratively build J_o while preventing reselection of rows, allowing for single row or block selection.

    Args:
        J_c: The comprehensive Jacobian matrix of shape (n, m).
        num_rows: Number of rows to select in total.
        n_freq: Number of frequency rows per source-receiver pair.
        n_receivers: Number of receivers per source.
        selection_mode: "single" for single row selection, "block" for source-wise block selection.
        threshold: Eigenvalue criterion threshold.

    Returns:
        J_o_final: The selected rows forming the optimized Jacobian.
        O_final: Selection vector marking selected rows.
        mask_history: History of active masks during iterations.
        criterion_log: Log of criterion improvements for each row or block.
    """
    n, m = J_c.shape

    def step_fn(carry, _):
        J_o, O, mask, mask_history, criterion_log, selection_mask = carry

        jax.debug.print("Iteration: {}, Sources Selected: {}", _, mask.sum())

        # Masked J_o to consider only active rows
        active_J_o = J_o * mask[:, None]

        # Current criterion based on active rows
        current_criterion = jax.lax.cond(
            mask.sum() > 0,
            lambda _: eigenvalue_criterion(active_J_o, threshold),
            lambda _: 0.0,
            operand=None,
        )

        if selection_mode == "single":
            def compute_score(row):
                zero_row_idx = jnp.argmax(jnp.all(active_J_o == 0, axis=1))
                augmented_J_o = active_J_o.at[zero_row_idx].set(row)
                return eigenvalue_criterion(augmented_J_o, threshold) - current_criterion

            scores = jax.vmap(compute_score)(J_c)

        elif selection_mode == "block":
            def compute_block_score(block_start):
                zero_row_idx = jnp.argmax(jnp.all(active_J_o == 0, axis=1))
                block_size = n_freq * n_receivers
                # block_size = jnp.where(block_start + block_size > n, n - block_start, block_size)
                block = lax.dynamic_slice(J_c, (block_start, 0), (block_size, m))
                # block = J_c[block_start:block_start + block_size]
                augmented_J_o = lax.dynamic_update_slice(active_J_o, block, (zero_row_idx, 0))
                # augmented_J_o = active_J_o.at[zero_row_idx:zero_row_idx + block_size].set(block)
                return eigenvalue_criterion(augmented_J_o, threshold) - current_criterion

            block_starts = jnp.arange(0, n, n_freq * n_receivers)
            scores = jax.vmap(compute_block_score)(block_starts)
        else:
            raise ValueError("Invalid selection_mode. Use 'single' or 'block'.")

        # Apply the selection mask: Set scores of already selected rows or blocks to -inf
        scores = jnp.where(selection_mask == 1, scores, -jnp.inf)

        # Log the scores (criterion improvements)
        criterion_log = criterion_log.at[_, :].set(scores)

        # Select the best row or block (hard selection)
        best_idx = jnp.argmax(scores)

        if selection_mode == "single":
            best_row_idx = best_idx
            best_row = J_c[best_row_idx]
            idx = jnp.argmax(mask == 0)
            J_o = J_o.at[idx].set(best_row.astype(jnp.complex64))
            mask = mask.at[idx].set(1)
            O = O.at[best_row_idx].set(1)
            selection_mask = selection_mask.at[best_row_idx].set(0)
        elif selection_mode == "block":
            block_start = best_idx * n_freq * n_receivers
            block_size = n_freq * n_receivers
            # block_size = jnp.where(block_start + block_size > n, n - block_start, block_size)
            block = lax.dynamic_slice(J_c, (block_start, 0), (block_size, m))
            # idx = jnp.argmax(mask == 0)
            # J_o = lax.dynamic_update_slice(J_o, block.astype(jnp.complex64), (idx, 0))
            # block = J_c[block_start:block_start + block_size]
            idx = jnp.argmax(mask == 0)
            
            J_o = lax.dynamic_update_slice(J_o, block.astype(jnp.complex64), (idx, 0))

            # J_o = J_o.at[idx:idx + block_size].set(block)
            mask_update = jnp.ones((block_size,), dtype=jnp.complex64)
            mask = lax.dynamic_update_slice(mask, mask_update, (idx,))            
            # O = O.at[block_start:block_start + block_size].set(1)
            # selection_mask = selection_mask.at[block_start:block_start + block_size].set(0)
            mask_update = jnp.ones((block_size,), dtype=jnp.complex64)
            mask = lax.dynamic_update_slice(mask, mask_update, (idx,))

            O_update = jnp.ones((block_size, 1), dtype=jnp.complex64)
            O = lax.dynamic_update_slice(O, O_update, (block_start, 0))

            # selection_mask_update = jnp.zeros((block_size,), dtype=jnp.complex64)
            # selection_mask = lax.dynamic_update_slice(selection_mask, selection_mask_update, (block_start,))
            block_idx = best_idx  # Use `best_idx` as block index
            selection_mask_update = jnp.zeros((1,), dtype=jnp.complex64)  # Update single block
            selection_mask = lax.dynamic_update_slice(selection_mask, selection_mask_update, (block_idx,))
        # Append the current mask to mask_history
        mask_history = mask_history.at[_, :].set(mask)

        return (J_o, O, mask, mask_history, criterion_log, selection_mask), None
    
    
    # Initialize J_o, O, mask, and logging arrays based on selection_mode
    if selection_mode == "single":
        J_o_init = jnp.zeros((num_rows, m), dtype=jnp.complex64)  # Pre-allocate J_o with a fixed shape
        O_init = jnp.zeros((n, 1), dtype=jnp.complex64)  # Initial selection vector
        mask_init = jnp.zeros(num_rows, dtype=jnp.complex64)  # Mask to track active rows in J_o
        mask_history_init = jnp.zeros((num_rows, num_rows), dtype=jnp.complex64)  # Store masks at each iteration
        criterion_log_init = jnp.zeros((num_rows, n), dtype=jnp.complex64)  # Log criterion improvements for each row
        selection_mask_init = jnp.ones(n, dtype=jnp.complex64)  # Selection mask (1 = eligible, 0 = selected)
    elif selection_mode == "block":
        num_blocks = len(jnp.arange(0, n, n_freq * n_receivers))
        J_o_init = jnp.zeros((num_rows * n_freq * n_receivers, m), dtype=jnp.complex64)  # Adjust for block sizes
        O_init = jnp.zeros((n, 1), dtype=jnp.complex64)  # Initial selection vector
        mask_init = jnp.zeros(num_rows * n_freq * n_receivers, dtype=jnp.complex64)  # Mask to track active rows in J_o
        mask_history_init = jnp.zeros((num_rows * n_freq * n_receivers, num_rows * n_freq * n_receivers), dtype=jnp.complex64)  # Store masks at each iteration
        criterion_log_init = jnp.zeros((num_rows, num_blocks), dtype=jnp.complex64)  # Log criterion improvements for each block
        selection_mask_init = jnp.ones(num_blocks, dtype=jnp.complex64)  # Selection mask (1 = eligible, 0 = selected)
    else:
        raise ValueError("Invalid selection_mode. Use 'single' or 'block'.")

    # Iterate using JAX scan
    (J_o_final, O_final, _, mask_history, criterion_log, _), _ = scan(
        step_fn, 
        (J_o_init, O_init, mask_init, mask_history_init, criterion_log_init, selection_mask_init), 
        jnp.arange(num_rows)
    )
    return J_o_final, O_final, mask_history, criterion_log

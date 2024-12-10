import jax
import jax.numpy as jnp
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


def iterative_selection_no_reselection(J_c, num_rows, threshold=0.5):
    """Iteratively build J_o while preventing reselection of rows."""
    n, m = J_c.shape

    def step_fn(carry, _):
        J_o, O, mask, mask_history, criterion_log, selection_mask = carry

        # Masked J_o to consider only active rows
        active_J_o = J_o * mask[:, None]

        # Current criterion based on active rows
        current_criterion = cond(
            mask.sum() > 0,
            lambda _: eigenvalue_criterion(active_J_o, threshold),
            lambda _: 0.0,
            operand=None,
        )

        # Compute scores by evaluating criterion improvement for each row
        def compute_score(row):
            # Find the first zero row in active_J_o
            zero_row_idx = jnp.argmax(jnp.all(active_J_o == 0, axis=1))

            # Replace the zero row with the candidate row
            augmented_J_o = active_J_o.at[zero_row_idx].set(row)

            # Compute the eigenvalue criterion for the updated J_o
            return eigenvalue_criterion(augmented_J_o, threshold) - current_criterion


        scores = jax.vmap(compute_score)(J_c)

        # Apply the selection mask: Set scores of already selected rows to -inf
        scores = jnp.where(selection_mask == 1, scores, -jnp.inf)

        # Log the scores (criterion improvements)
        criterion_log = criterion_log.at[_, :].set(scores)

        # Select the best row (hard selection)
        best_row_idx = jnp.argmax(scores)
        best_row = J_c[best_row_idx]

        # Update J_o using the next available row slot
        idx = jnp.argmax(mask == 0)  # Find the next available row
        J_o = J_o.at[idx].set(best_row.astype(jnp.complex64))

        # Update the mask to mark the new row as active
        mask = mask.at[idx].set(1)

        # Update the selection vector O (mark the selected row only once)
        O = O.at[best_row_idx].set(1)

        # Update the selection mask to exclude the selected row
        selection_mask = selection_mask.at[best_row_idx].set(0)

        # Append the current mask to mask_history
        mask_history = mask_history.at[_, :].set(mask)

        return (J_o, O, mask, mask_history, criterion_log, selection_mask), None

    # Initialize J_o, O, mask, and logging arrays
    J_o_init = jnp.zeros((num_rows, m), dtype=jnp.complex64)  # Pre-allocate J_o with a fixed shape
    O_init = jnp.zeros((n, 1), dtype=jnp.complex64)  # Initial selection vector
    mask_init = jnp.zeros(num_rows, dtype=jnp.complex64)  # Mask to track active rows in J_o
    mask_history_init = jnp.zeros((num_rows, num_rows), dtype=jnp.complex64)  # Store masks at each iteration
    criterion_log_init = jnp.zeros((num_rows, n), dtype=jnp.complex64)  # Log criterion improvements for each row
    selection_mask_init = jnp.ones(n, dtype=jnp.complex64)  # Selection mask (1 = eligible, 0 = selected)

    # Iterate using JAX scan
    (J_o_final, O_final, _, mask_history, criterion_log, _), _ = scan(
        step_fn, 
        (J_o_init, O_init, mask_init, mask_history_init, criterion_log_init, selection_mask_init), 
        jnp.arange(num_rows)
    )
    return J_o_final, O_final, mask_history, criterion_log
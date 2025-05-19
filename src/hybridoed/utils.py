from matplotlib import pyplot as plt


def plot_specific_jacobian(jacobians, frequencies, src_coords, rec_coords, 
                           source_idx=None, receiver_idx=None, frequency_idx=None, save_path=None):
    """
    Plot the real and imaginary parts of a specific Jacobian from a structured array.

    Parameters:
        jacobians (array): Array of shape (n_sources, n_receivers, n_frequencies, rows, cols).
        frequencies (list): List of frequencies corresponding to the third dimension of `jacobians`.
        src_coords (list): List of source coordinates.
        rec_coords (array): Array of receiver coordinates.
        source_idx (int, optional): Index of the source to plot.
        receiver_idx (int, optional): Index of the receiver to plot.
        frequency_idx (int, optional): Index of the frequency to plot.
        save_path (str, optional): Path to save the plots. If None, displays them instead.
    """
    if source_idx is None or receiver_idx is None or frequency_idx is None:
        raise ValueError("You must specify source_idx, receiver_idx, and frequency_idx.")

    # Extract real and imaginary parts
    real_part = jacobians[source_idx, receiver_idx, frequency_idx, :].reshape(70,140)[:,:70]
    imaginary_part = jacobians[source_idx, receiver_idx, frequency_idx, :].reshape(70,140)[:,70:]

    # Plot real part
    plt.figure(figsize=(10, 8))
    plt.imshow(real_part, aspect='auto', cmap='viridis')
    plt.colorbar(label="Magnitude")
    title_real = (
        f"Jacobian Real Part\n"
        f"Frequency: {frequencies[frequency_idx]} Hz, "
        f"Source: {src_coords[source_idx][0]}, "
        f"Receiver: {rec_coords[receiver_idx]}"
    )
    plt.title(title_real)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    # Overlay source and receiver markers
    if src_coords is not None and rec_coords is not None:
        plt.scatter(src_coords[source_idx][1], src_coords[source_idx][0], marker='*', c='red', s=100)
        plt.scatter(rec_coords[receiver_idx, 1], rec_coords[receiver_idx, 0], marker='^', c='blue', s=100)

    # Save or show real part
    if save_path:
        filename_real = f"{save_path}/jacobian_real_f{frequency_idx}_s{source_idx}_r{receiver_idx}.png"
        plt.savefig(filename_real, dpi=300)
    else:
        plt.show()

    # Plot imaginary part
    plt.figure(figsize=(10, 8))
    plt.imshow(imaginary_part, aspect='auto', cmap='viridis')
    plt.colorbar(label="Magnitude")
    title_imag = (
        f"Jacobian Imaginary Part\n"
        f"Frequency: {frequencies[frequency_idx]} Hz, "
        f"Source: {src_coords[source_idx][0]}, "
        f"Receiver: {rec_coords[receiver_idx]}"
    )
    plt.title(title_imag)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()

    # Overlay source and receiver markers
    if src_coords is not None and rec_coords is not None:
        plt.scatter(src_coords[source_idx][1], src_coords[source_idx][0], marker='*', c='red', s=100)
        plt.scatter(rec_coords[receiver_idx, 1], rec_coords[receiver_idx, 0], marker='^', c='blue', s=100)

    # Save or show imaginary part
    if save_path:
        filename_imag = f"{save_path}/jacobian_imag_f{frequency_idx}_s{source_idx}_r{receiver_idx}.png"
        plt.savefig(filename_imag, dpi=300)
    else:
        plt.show()
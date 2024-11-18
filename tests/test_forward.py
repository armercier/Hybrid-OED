import jax.numpy as jnp
import pytest
from hybridoed.forward import create_src_field
from jwave.geometry import Domain

# Mock FourierSeries function
# def FourierSeries(array, domain):
#     """Mock FourierSeries function for testing purposes."""
#     return jnp.fft.fft(array, axis=0)

# # Replace the actual FourierSeries in the module for testing
# import project_name.core
# project_name.core.FourierSeries = FourierSeries

def test_create_src_field():
    # Parameters
    N = (10, 10)  # Grid size
    dx = (1.0, 1.0)  # Grid spacing
    x, y = 5.3, 7.7  # Source coordinates (non-integer)
    # domain = jnp.linspace(0, 1, 10)  # Domain for Fourier Series
    domain = Domain(N, dx)
    omega = 2.0  # Scaling factor

    # Call the function
    src = create_src_field(N, x, y, domain, omega)

    # Assertions
    assert src.domain.grid.shape == (10, 10,2), "Output shape should match the domain size"
    assert src.is_complex, "Output should be a complex array"

def test_boundary_conditions():
    N = (10, 10)
    dx = (1.0, 1.0)  # Grid spacing

    # domain = jnp.linspace(0, 1, 10)
    domain = Domain(N, dx)
    omega = 1.0

    # Test at boundaries
    src_lower = create_src_field(N, 0, 0, domain, omega)
    src_upper = create_src_field(N, 9, 9, domain, omega)

    # Assertions
    assert src_lower.is_complex, "Lower boundary should produce a valid complex array"
    assert src_upper.is_complex, "Upper boundary should produce a valid complex array"

def test_invalid_input():
    N = (10, 10)
    dx = (1.0, 1.0)  # Grid spacing
    # domain = jnp.linspace(0, 1, 10)
    domain = Domain(N, dx)
    omega = 1.0

    with pytest.raises(ValueError):
        # Invalid domain size (mismatch with FourierSeries requirements)
        invalid_domain = jnp.linspace(0, 1, 5)
        create_src_field(N, 5, 5, invalid_domain, omega)
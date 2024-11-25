import jax.numpy as jnp
from hybridoed.forward import create_src_field, solve_helmholtz_general
from jwave.geometry import Domain


# write a test taht compute a forward model and compare it ot the file true_field.npy
def test_wavefield_solve():
    N = (64, 64) 
    dx = (1.0, 1.0)  # Spatial resolution
    f = 200
    omega = 2 * jnp.pi * f
    domain = Domain(N, dx)

    # source position
    x = N[0]//2 + 0.01
    y = N[1]//2 + 0.01

    sound_speed = jnp.full(N, 1540.)
    pml_size = 15


    true_field = solve_helmholtz_general(sound_speed, domain=domain, src_coord=(y,x), omega=omega, pml_size=pml_size)

    # read true_field from file
    # get the relative path to the file



    true_field_file = jnp.load("tests/integration/true_field.npy")

    # compare the two fields
    assert jnp.allclose(true_field, true_field_file, atol=1e-1), "The computed field does not match the true field"


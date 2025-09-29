import pytest
import numpy as np
import jax
import jax.numpy as jnp

# --- Imports under test -------------------------------------------------------
# Adjust paths/names if you moved things.
from exogibbs.presets.ykb4 import chemsetup
from exogibbs.api.chemistry import ChemicalSetup


def test_prepare_ykb4_setup_basic():
    """Builds ChemicalSetup without raising, and fields look sane."""
    setup = setup()
    assert isinstance(setup, ChemicalSetup)

    # core
    assert isinstance(setup.formula_matrix, jnp.ndarray)
    assert callable(setup.hvector_func)
    E, K = setup.formula_matrix.shape
    assert E > 0 and K > 0

    # optional metadata
    assert (setup.elements is None) or isinstance(setup.elements, tuple)
    assert (setup.species is None) or isinstance(setup.species, tuple)
    assert (setup.element_vector_reference is None) or isinstance(
        setup.element_vector_reference, (np.ndarray, jnp.ndarray)
    )
    assert (setup.metadata is None) or isinstance(setup.metadata, dict)


def test_hvector_func_shape_and_types():
    """hvector_func(T) returns (K,) and is JAX-friendly."""
    setup = setup()
    K = setup.formula_matrix.shape[1]

    # scalar T
    out = setup.hvector_func(500.0)
    assert isinstance(out, jnp.ndarray)
    assert out.shape == (K,)

    # batched T via vmap
    Ts = jnp.array([400.0, 500.0, 600.0])
    batched = jax.vmap(setup.hvector_func)(Ts)  # (B, K)
    assert batched.shape == (Ts.shape[0], K)


def test_hvector_func_grad_and_jit():
    """grad/jit should work through T."""
    setup = setup()

    def f(T):
        return jnp.sum(setup.hvector_func(T))

    # grad at a scalar temperature
    g = jax.grad(f)(900.0)
    assert jnp.isfinite(g)

    # jit-compiled call
    f_jit = jax.jit(f)
    val = f_jit(550.0)
    assert jnp.isfinite(val)


def test_dimension_consistency():
    """K (species dim) consistent between formula_matrix and h(T)."""
    setup = setup()
    K = setup.formula_matrix.shape[1]
    out = setup.hvector_func(300.0)
    assert out.shape[-1] == K


def test_optional_b_reference_host_side():
    """element_vector_reference is optional and (if present) host-side array."""
    setup = setup()
    b_ref = setup.element_vector_reference
    if b_ref is not None:
        # Treat as reference only; ensure it’s not an unexpectedly huge DeviceArray
        assert isinstance(b_ref, (np.ndarray, jnp.ndarray))
        assert b_ref.ndim == 1 and b_ref.size == setup.formula_matrix.shape[0]
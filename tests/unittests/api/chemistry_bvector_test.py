import jax
import jax.numpy as jnp

from exogibbs.api.chemistry import ChemicalSetup, update_element_vector, element_indices_by_name


def test_update_b_vector_and_indices_jit_safe():
    # Elements order (includes electron as last entry)
    elements = ("C", "H", "He", "K", "N", "Na", "O", "P", "S", "Ti", "V", "e-")
    K = len(elements)

    # Reference abundances (arbitrary positive values)
    b_ref = jnp.array([1.0, 10.0, 0.5, 0.01, 1.2, 0.02, 2.0, 0.03, 0.04, 0.005, 0.006, 0.0])

    setup = ChemicalSetup(
        formula_matrix=jnp.zeros((K, 1)),  # not used here
        hvector_func=lambda T: jnp.zeros((1,)),  # not used here
        elements=elements,
    )

    idx_COe = element_indices_by_name(setup, ["C", "O", "e-"])
    idx_C, idx_O, idx_e = int(idx_COe[0]), int(idx_COe[1]), int(idx_COe[2])

    # Scales for C and O; set electron to 0.0
    def make_b(C_scale, O_scale):
        return update_element_vector(
            b_ref,
            scale_indices=jnp.array([idx_C, idx_O]),
            scales=jnp.array([C_scale, O_scale]),
            set_indices=jnp.array([idx_e]),
            set_values=jnp.array([0.0]),
        )

    # JIT compile to ensure trace-safety
    make_b_jit = jax.jit(make_b)

    C_scale = 0.8
    O_scale = 1.1
    out = make_b_jit(C_scale, O_scale)

    # Expected: only C and O scaled; e- set to 0; others unchanged
    expected = b_ref.at[idx_C].set(b_ref[idx_C] * C_scale)
    expected = expected.at[idx_O].set(b_ref[idx_O] * O_scale)
    expected = expected.at[idx_e].set(0.0)

    assert jnp.allclose(out, expected)

from exogibbs.optimize.core import _A_diagn_At

def test_A_diag_At():
    """ Test the _A_diag_At function to ensure it computes the expected result.
    This function computes A diag(n) A^T using jnp.einsum.
    
    """
    import jax.numpy as jnp
    number_density_vector = jnp.array([1.0, 2.0, 3.0])
    formula_matrix = jnp.array([[1.0, 0.0, 1.0],
                                [0.0, 1.0, 2.0],
                                [1.0, 2.0, 3.0]])
    
    result = _A_diagn_At(number_density_vector, formula_matrix)
    expected_result = formula_matrix @ jnp.diag(number_density_vector) @ formula_matrix.T
    
    assert jnp.allclose(result, expected_result), f"Expected {expected_result}, got {result}"


if __name__ == "__main__":
    test_A_diag_At()
    print("Test passed!")
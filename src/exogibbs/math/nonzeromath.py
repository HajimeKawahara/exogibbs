import numpy as np
import jax.numpy as jnp

def np_nonzeromax(x, A, fill_value=-np.inf):
    """computes m_j = max_{i : A_ji ne 0} x_i

    Args:
        x: 1D array, shape (n,)
        A: 2D array, shape (m,n)
        fill_value: float, value to use when all A_ji are zero for a given j

    Returns:
        m: 1D array, shape (m,)

    """
    masked_A = A != 0.0  # shape (m,n)
    x_broadcasted = np.broadcast_to(x, A.shape)  # shape (m,n)
    x_masked = np.where(masked_A, x_broadcasted, -np.inf)  # shape (m,n)
    m = np.max(x_masked, axis=1)  # shape (m,)
    m = np.where(m == -np.inf, fill_value, m)
    return m
    
def nonzeromax(x, masked_A):
    """computes m_j = max_{i : A_ji ne 0} x_i, assuming A_ij >= 0, returns m_j with fill_value is 0.0
    Args:
        x: 1D jax array, shape (n,)
        masked_A: 2D jax boolean array, shape (m,n), True where A_ji ne 0, can be obtained by masked_A = jnp.array(A != 0.0)

    Returns:
        m: 1D jax array, shape (m,)

    """
    return jnp.max(x*masked_A, axis=1)


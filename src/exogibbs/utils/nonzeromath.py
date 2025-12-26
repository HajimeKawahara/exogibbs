import numpy as np


def nonzeromax(x, A, fill_value=-np.inf):
    """computes m_j = max_{i : A_ji ne 0} x_i

    Args:
        x: 1D array, shape (n,)
        A: 2D array, shape (m,n)
        fill_value: float, value to use when all A_ji are zero for a given j

    Returns:
        m: 1D array, shape (m,)

    """
    mask_vector = A != 0.0  # shape (m,n)
    x_broadcasted = np.broadcast_to(x, A.shape)  # shape (m,n)
    x_masked = np.where(mask_vector, x_broadcasted, -np.inf)  # shape (m,n)
    m = np.max(x_masked, axis=1)  # shape (m,)
    m = np.where(m == -np.inf, fill_value, m)
    return m

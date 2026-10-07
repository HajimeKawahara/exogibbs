"""max over nonzero matrix component


"""

import numpy as np
import jax.numpy as jnp
from exogibbs.math.nonzeromath import np_nonzeromax
from exogibbs.math.nonzeromath import nonzeromax

def test_np_nonzeromax():
    A = np.array([[1.0, 0.0, 2.0],
                   [0.0, 0.0, 3.0],
                   [4.0, 5.0, 0.0],
                   [0.0, 0.0, 0.0]])
    x = np.array([10.0, 20.0, 30.0])
    m = np_nonzeromax(x, A, fill_value=0.0)
    assert np.allclose(m, np.array([30.0, 30.0, 20.0, 0.0]))


def test_np_nonzeromax_preserves_selected_negative_infinity():
    result = np_nonzeromax(
        np.array([-np.inf, -2.0]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        fill_value=0.0,
    )

    assert np.array_equal(result, np.array([-np.inf, 0.0]))

def test_jnp_nonzeromax():
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    A = np.array([[0.0, 1.0, 0.0, 2.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [5.0, 0.0, 6.0, 0.0]])
    m = np_nonzeromax(x, A, fill_value=0.0)

    masked_A = jnp.array(A != 0.0)
    mj = nonzeromax(x, masked_A)
    assert np.all(m == np.array(mj))


def test_jnp_nonzeromax_ignores_masked_values_for_negative_inputs():
    x = jnp.array([-3.0, -2.0, -1.0])
    masked_A = jnp.array(
        [
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ]
    )

    result = nonzeromax(x, masked_A)

    assert np.allclose(result, np.array([-3.0, -2.0, 0.0]))

if __name__ == "__main__":
    test_np_nonzeromax()
    test_jnp_nonzeromax()

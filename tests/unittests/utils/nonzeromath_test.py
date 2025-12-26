"""max over nonzero matrix component


"""

import numpy as np
import jax.numpy as jnp
from exogibbs.utils.nonzeromath import np_nonzeromax
from exogibbs.utils.nonzeromath import jnp_nonzeromax

def test_np_nonzeromax():
    A = np.array([[1.0, 0.0, 2.0],
                   [0.0, 0.0, 3.0],
                   [4.0, 5.0, 0.0],
                   [0.0, 0.0, 0.0]])
    x = np.array([10.0, 20.0, 30.0])
    m = np_nonzeromax(x, A, fill_value=0.0)
    assert np.allclose(m, np.array([30.0, 30.0, 20.0, 0.0]))

def test_jnp_nonzeromax():
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    A = np.array([[0.0, 1.0, 0.0, 2.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [5.0, 0.0, 6.0, 0.0]])
    m = np_nonzeromax(x, A, fill_value=0.0)
    
    masked_A = jnp.array(A != 0.0)
    mj = jnp_nonzeromax(x, masked_A)
    assert np.all(m == np.array(mj))

if __name__ == "__main__":
    test_np_nonzeromax()
    test_jnp_nonzeromax()
    
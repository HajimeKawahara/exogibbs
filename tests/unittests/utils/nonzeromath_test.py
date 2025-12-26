"""max over nonzero matrix component


"""

import numpy as np
from exogibbs.utils.nonzeromath import nonzeromax

def test_nonzeromax():
    A = np.array([[1.0, 0.0, 2.0],
                   [0.0, 0.0, 3.0],
                   [4.0, 5.0, 0.0],
                   [0.0, 0.0, 0.0]])
    x = np.array([10.0, 20.0, 30.0])
    m = nonzeromax(x, A, fill_value=0.0)
    assert np.allclose(m, np.array([30.0, 30.0, 20.0, 0.0]))

if __name__ == "__main__":
    test_nonzeromax()
    
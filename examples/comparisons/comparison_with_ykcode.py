"""
Validation of Gibbs Minimization Against ykawashima's B4 code
============================================================

This example demonstrates and validates the ExoGibbs thermochemical equilibrium
solver against the code by ykawashima when she was at B4.

Updated to use the high-level API: exogibbs.api.equilibrium.equilibrium.
"""

from exogibbs.presets.ykb4 import prepare_ykb4_setup
from exogibbs.api.equilibrium import equilibrium, EquilibriumOptions
import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


# Thermodynamic conditions
temperature = 500.0  # K
P = 10.0  # bar
Pref = 1.0  # bar, reference pressure

# chemical setup
chem = prepare_ykb4_setup()

# old reference elemental abundance b from yk's sample number densities
b_old_ref = jnp.array(
    [
        4.8774824e-04,
        1.6749767e00,
        1.6143440e-01,
        2.5438149e-07,
        1.3435642e-04,
        3.9624806e-06,
        9.7356725e-04,
        5.7690579e-07,
        3.0653933e-05,
        1.6687756e-07,
        1.9870969e-08,
        0.0000000e00,
    ]
)


##############################################################################
# Solve equilibrium via high-level API
# ------------------------------------
opts = EquilibriumOptions(epsilon_crit=1e-11, max_iter=1000)
res = equilibrium(
    chem,
    T=temperature,
    P=P,
    b=b_old_ref,
    Pref=Pref,
    options=opts,
)

##############################################################################
nk_result = res.n

# load yk's results for 10 bar
dat = np.loadtxt("../data/p10.txt", delimiter=",")
mask = dat > 1.0e-14
mask_nk_result = nk_result[mask]
mask_dat = dat[mask]

res = mask_nk_result / mask_dat - 1.0
print(res, "diff for n>1.e-14")
assert np.max(np.abs(res)) < 0.051
# 8/9/2025
# [-0.00163185 -0.00163185  0.02571018 -0.00203837 -0.05069541 -0.00163185
# -0.00481986 -0.00420364 -0.00161074 -0.00163182 -0.00163185 -0.00163183
# -0.00163184 -0.00163178 -0.00163185 -0.00163184]


ind = np.arange(len(nk_result))
import matplotlib.pyplot as plt

plt.plot(ind, nk_result, "+", label="ExoGibbs")
plt.plot(ind, dat, ".", alpha=0.5, label="yk B4 code")
plt.xlabel("Species Index")
plt.ylabel("Number (log scale)")
plt.yscale("log")
plt.legend()
plt.grid()
plt.show()

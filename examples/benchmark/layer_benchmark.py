from arrow import get
from exogibbs.presets.ykb4 import prepare_ykb4_setup
from exogibbs.api.equilibrium import equilibrium_profile, EquilibriumOptions, EquilibriumInit
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm  
from numpy.random import default_rng
from numpy.random import normal
import numpy as np
import time
# chemical setup
chem = prepare_ykb4_setup()
b_element_vector = chem.b_element_vector_reference

# Thermodynamic conditions
Pref = 1.0  # bar, reference pressure
opts = EquilibriumOptions(epsilon_crit=1e-11, max_iter=10000)

Nlayer = 1000
Nsample = 100
#Parr = jnp.logspace(2, -8, Nlayer)  # Pressure from 1 bar to 1e-6 bar
Parr = jnp.logspace(-3, -4, Nlayer)  # Pressure from 1 bar to 1e-6 bar


### optimized initialization
Tarr = (Parr)**0.02 * 1000.0
res = equilibrium_profile(
    chem,
    Tarr,
    Parr,
    b_element_vector,
    Pref=Pref,
    options=opts,
    )

#ln_n_init = res.ln_n  # (Nlayer, K)
#ln_ntot_init = jnp.log(jnp.sum(res.n, axis=1))  # (Nlayer,)

opts = EquilibriumOptions(epsilon_crit=1e-11, max_iter=1000)

# applies JIT to equilibrium_profile, this makes code faster
@jit
def get_res(Tarr,Parr):
    return equilibrium_profile(
        chem,
        Tarr,
        Parr,
        b_element_vector,
        Pref=Pref,
        #init=EquilibriumInit(ln_nk=ln_n_init, ln_ntot=ln_ntot_init),
        options=opts,
        )

# benchmark loop
rng = default_rng(0)
tstart = time.time()
for isample in tqdm(range(Nsample)):

    alpha = normal(0,0.01) + 0.02
    Tarr = (Parr)**alpha * 1000.0
    Tarr = jnp.clip(Tarr, 200.0, 1500.0)
    res = get_res(Tarr,Parr)

    
tend = time.time()
print(f"Total time for {Nsample} samples and {Nlayer} layers: {tend - tstart:.2f} seconds")
print(f"Average time per sample: {(tend - tstart)/Nsample:.4f} seconds")
print(f"Average time per layer: {(tend - tstart)/(Nsample*Nlayer):.6f} seconds")

### Benchmark Results for default init
#Total time for 100 samples and 100 layers: 149.91 seconds
#Average time per sample: 1.4991 seconds
#Average time per layer: 0.014991 seconds

#JIT
#Total time for 100 samples and 1000 layers: 19.64 seconds
#Average time per sample: 0.1964 seconds
#Average time per layer: 0.000196 seconds
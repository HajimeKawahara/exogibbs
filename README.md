# ExoGibbs
 [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HajimeKawahara/exogibbs)

Differentiable Thermochemical Equilibrium, powered by JAX. 

The optimization scheme is based on the Lagrange multiplier, similar to [NASA/CEA algorithm](https://ntrs.nasa.gov/api/citations/19950013764/downloads/19950013764.pdf). 
The terminology follows Smith and Missen, [Chemical Reaction Equilibrium Analysis](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690310127) (1983, Wiley-Interscience). 

## Basic Use

```python
from exogibbs.presets.ykb4 import prepare_ykb4_setup
from exogibbs.api.equilibrium import equilibrium_profile, EquilibriumOptions

# chemical setup
chem = prepare_ykb4_setup()

# Thermodynamic conditions
Pref = 1.0  # bar, reference pressure
opts = EquilibriumOptions(epsilon_crit=1e-11, max_iter=1000)

res = equilibrium_profile(
    chem,
    temperature_profile,
    pressure_profile,
    chem.element_vector,
    Pref=Pref,
    options=opts,
)
nk_result = res.x #mixing ratio
```

ExoGibbs is designed to plug into [ExoJAX](https://github.com/HajimeKawahara/exojax) and enable gradient-based equilibrium retrievals. 
It is still in a very beta stage, so please use it at your own risk.

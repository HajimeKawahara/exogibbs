# ExoGibbs
 [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HajimeKawahara/exogibbs)

Differentiable Thermochemical Equilibrium, powered by JAX. 

The optimization scheme is based on the Lagrange multiplier, similar to [NASA/CEA algorithm](https://ntrs.nasa.gov/api/citations/19950013764/downloads/19950013764.pdf). 
The terminology follows Smith and Missen, [Chemical Reaction Equilibrium Analysis](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690310127) (1983, Wiley-Interscience). 

## Basic Use

```python
from jax import config
config.update("jax_enable_x64", True)

from exogibbs.presets.ykb4 import chemsetup
from exogibbs.api.equilibrium import equilibrium_profile, EquilibriumOptions

# chemical setup
chem = chemsetup()

# Thermodynamic conditions
opts = EquilibriumOptions(epsilon_crit=1e-15, max_iter=1000)

res = equilibrium_profile(
    chem,
    temperature_profile,
    pressure_profile,
    chem.element_vector_reference,
    Pref=1.0,
    options=opts,
)
nk_result = res.x #mixing ratio
```

## presets

- ykb4: number of species: 160     elements: 12
- fastchem: number of species: 523    elements: 28


ExoGibbs is designed to plug into [ExoJAX](https://github.com/HajimeKawahara/exojax) and enable gradient-based equilibrium retrievals. 
It is still in a beta stage, so please use it at your own risk.


This package bundles logK data from [FastChem](https://github.com/NewStrangeWorlds/FastChem) in `fastchem` presets,
which is distributed under the GNU General Public License v3 (GPLv3).
Accordingly, ExoGibbs is also distributed under the GPLv3 license.
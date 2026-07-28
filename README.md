# ExoGibbs
 [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HajimeKawahara/exogibbs)

Differentiable Thermochemical Equilibrium, powered by JAX.

The optimization scheme is based on the Lagrange multiplier, similar to [NASA/CEA algorithm](https://ntrs.nasa.gov/api/citations/19950013764/downloads/19950013764.pdf).
The terminology follows Smith and Missen, [Chemical Reaction Equilibrium Analysis](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.690310127) (1983, Wiley-Interscience).

## Basic Use

```python
from jax import config
config.update("jax_enable_x64", True)

from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets.ykb4 import chemsetup

chem = chemsetup()
opts = EquilibriumOptions(epsilon_crit=1e-15, max_iter=1000)
res = solve_profile(
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

## v0.4 validation milestone

An independent four-point comparison with FastChem 4.0.3 found identical
major-gas species sets at 0.1 bar and 1200, 1400, 1600, and 1800 K, with
convergence and elemental-budget closure in both solvers. This supports the
scoped v0.4 major-gas milestone for the gas phase of the production
gas-plus-condensate solver. Detailed condensate phase selection remains a
v0.5 validation target.

See the
[v0.4 FastChem validation demo](https://github.com/HajimeKawahara/exogibbs/blob/main/documents/v0_4_fastchem4_validation_demo.rst)
and the
[technical comparison protocol](https://github.com/HajimeKawahara/exogibbs/blob/main/documents/fastchem4_production_comparison.rst).
Readable Python plots are provided for
[gas-only](https://github.com/HajimeKawahara/exogibbs/blob/main/examples/comparisons/comparison_with_fastchem4_gas.py)
and
[gas-plus-condensate](https://github.com/HajimeKawahara/exogibbs/blob/main/examples/comparisons/comparison_with_fastchem4_condensates.py)
comparisons.
The
[comparison example lineage](https://github.com/HajimeKawahara/exogibbs/blob/main/documents/comparison_example_lineage.rst)
also records the restored grid-initializer, analytical H and H/C/O, and
historical YK B4 regression demonstrations without treating them as extra
v0.4 acceptance points.

ExoGibbs is designed to plug into [ExoJAX](https://github.com/HajimeKawahara/exojax) and enable gradient-based equilibrium retrievals.
It is still in a beta stage, so please use it at your own risk.


This package bundles logK data from [FastChem](https://github.com/NewStrangeWorlds/FastChem) in `fastchem` presets,
which is distributed under the GNU General Public License v3 (GPLv3).
Accordingly, ExoGibbs is also distributed under the GPLv3 license.

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
- fastchem4: number of gas species: 422    elements: 28
- fastchem4_cond: number of condensate species: 219

## v0.7 release highlights

Version 0.7 strengthens fail-closed condensate validation. Candidate solutions
with non-finite gas amounts or non-finite or negative condensate amounts are
rejected, and inactive-phase closure no longer accepts non-finite driving
values. FastChem-backed grids now convert external physical number densities
to the ExoGibbs element-inventory gauge before storing initialization amounts.
FastChem-backed grids created before v0.7 store the old physical-density gauge
and must be regenerated; they are not migrated automatically.

The native ExoGibbs solvers and packaged FastChem-format thermochemistry no
longer require `pyfastchem`. Install `exogibbs[fastchem]` only to generate grids
with the FastChem backend or to explicitly verify an ExoGibbs grid against
FastChem. FastChem verification is now opt-in.

Release comparison examples now fail with a nonzero status when their declared
numerical checks fail. Retrieval smoke runs also record sampler diagnostics and
reject divergent, stuck, incomplete, or non-finite chains.

## v0.6 release highlights

Version 0.6 strengthens condensate equilibrium with canonical amount-gauge
normalization, stricter zero-barrier closure and physical audits, and stable
solver shapes across profile support changes. Rainout and custom/grid
initializers now follow a consistent caller-gauge contract. Budget and
rainout diagnostic schemas advance to v2.

This release also adds validation examples for KCl/Na2S condensation,
forsterite-enstatite competition, and Fe-FeS local equilibrium versus
sequential rainout, together with timing tools for the documented examples
and repeated full-catalog L-dwarf profiles.

See the [condensate solver guide](https://github.com/HajimeKawahara/exogibbs/blob/main/documents/equilibrium_solvers.rst),
the [profile and rainout guide](https://github.com/HajimeKawahara/exogibbs/blob/main/documents/condensate_profile.rst),
and the [documented example benchmarks](https://github.com/HajimeKawahara/exogibbs/blob/main/benchmarks/documented_examples/README.md).

## v0.5 release highlights

Version 0.5 adds a custom VJP for fixed-support condensate equilibrium,
opt-in bottom-to-top rainout profiles, condensate grid initialization, and
end-to-end ExoJAX/NumPyro NUTS examples. The differentiable contract is local
to a fixed condensate support; support changes and rainout propagation remain
outside automatic differentiation.

See the [condensate solver guide](https://github.com/HajimeKawahara/exogibbs/blob/main/documents/equilibrium_solvers.rst),
the [profile and rainout guide](https://github.com/HajimeKawahara/exogibbs/blob/main/documents/condensate_profile.rst),
and the [Ito et al. (2025) validation](https://github.com/HajimeKawahara/exogibbs/blob/main/documents/ito_2025_rainout_comparison.rst).

## v0.4 validation milestone

An independent four-point comparison with FastChem 4.0.3 found identical
major-gas species sets at 0.1 bar and 1200, 1400, 1600, and 1800 K, with
convergence and elemental-budget closure in both solvers. This supports the
scoped v0.4 major-gas milestone for the gas phase of the production
gas-plus-condensate solver. Detailed condensate phase selection was outside
the scoped v0.4 claim.

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

## Documentation

Install the pinned documentation dependencies and build the HTML pages with:

```bash
python -m pip install -e ".[docs]"
./update_doc.sh
```

The generated API reference and HTML output are written under `documents/`
and are intentionally excluded from version control. CI also publishes the
HTML output as the `Documentation HTML` artifact.

ExoGibbs is designed to plug into [ExoJAX](https://github.com/HajimeKawahara/exojax) and enable gradient-based equilibrium retrievals.
It is still in a beta stage, so please use it at your own risk.


This package bundles equilibrium-constant and elemental-abundance data from
[FastChem](https://github.com/NewStrangeWorlds/FastChem) v3.1.3 and v4.0.3 in
the `fastchem` and `fastchem4` presets. FastChem is distributed under the GNU
General Public License v3 (GPLv3).
Accordingly, ExoGibbs is also distributed under the GPLv3 license.

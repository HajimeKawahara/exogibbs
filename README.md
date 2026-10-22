# ExoGibbs

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HajimeKawahara/exogibbs)

Differentiable thermochemical equilibrium, powered by JAX.

ExoGibbs minimizes Gibbs energy subject to elemental conservation. It supports
atmospheric chemistry and coupled phase equilibria, with derivatives for
gradient-based modeling and [ExoJAX](https://github.com/HajimeKawahara/exojax)
retrievals.

## Equilibrium modes

| Mode | Public API | Thermodynamic inputs |
| --- | --- | --- |
| Gas only | `exogibbs.api.gas` | Gas standard thermochemistry |
| Gas and pure condensates | `exogibbs.api.condensate` | Gas and pure-condensate standard thermochemistry |
| Multiphase, including solution phases | `exogibbs.api.multiphase` | Full extensive Gibbs energy for each phase |

The [solver guide](documents/equilibrium_solvers.rst) explains units, amount
conventions, and derivative support. Condensate and multiphase derivatives
apply locally on a fixed support; phase selection is outside differentiation.

## Install and run

```bash
python -m pip install exogibbs
```

Enable JAX float64 before constructing a chemical setup. This gas-only example
uses the [curated preset](documents/presets/curated.rst), evaluated on
1500–4000 K with ions included by default. Temperature is in K, pressure in bar,
and `result.x` contains mole fractions in `chem.species` order.

```python
import jax
jax.config.update("jax_enable_x64", True)

from exogibbs.api.gas import EquilibriumOptions, solve
from exogibbs.presets.curated import chemsetup

chem = chemsetup()
result, diagnostics = solve(
    chem, T=2500.0, P=1.0, b=chem.element_vector_reference,
    options=EquilibriumOptions(epsilon_crit=1e-13),
    return_diagnostics=True,
)
print("Converged:", bool(diagnostics["converged"]))
print("H2O mole fraction:", float(result.x[chem.species.index("H2O1")]))
```

Other packaged presets include [YKB4](documents/presets/ykb4.rst),
[FastChem 3](documents/presets/fastchem.rst), and
[FastChem 4 gas and condensates](documents/presets/fastchem4.rst).
Install `exogibbs[fastchem]` for the optional FastChem grid backend or comparison
tools; the native solvers and packaged data do not require `pyfastchem`.

## Learn more

- [Documentation](documents/index.rst) and [public API](documents/api.rst).
- [Condensate profiles and rainout](documents/condensate_profile.rst).
- [Multiphase guide](documents/multiphase.rst) and the illustrated
  [plagioclase exsolution tutorial](documents/plagioclase_exsolution.rst).
- [Model assumptions and evidence (MCEG)](documents/mceg/index.rst).
- **[v0.7.0 release notes and upgrade guidance](documents/release_notes.rst).**

To build the HTML manual from a checkout, follow the
[documentation build guide](documents/mceg/build.rst), then run
`./update_doc.sh`. Open `documents/_build/html/index.html`.
The [Japanese manual](https://github.com/HajimeKawahara/doc_ExoGibbs) is maintained
separately and included as the `doc_ExoGibbs` submodule.

## License

ExoGibbs is distributed under [GPL-3.0-or-later](LICENSE).
Bundled [FastChem](https://github.com/NewStrangeWorlds/FastChem) v3.1.3 and
v4.0.3 thermochemistry and abundance data retain their upstream attribution.

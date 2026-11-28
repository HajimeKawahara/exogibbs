# Planetlib adoption

ExoGibbs adopts three partial models from Planetlib
[`a83b3fb255c7a383c6a8ef502328266f4dd16bff`](https://github.com/HajimeKawahara/planetlib/tree/a83b3fb255c7a383c6a8ef502328266f4dd16bff).
The standalone [MCEG contract](contract.yaml) records their quantities and
symbolic equations with ExoGibbs-owned implementation subjects, contexts and
pending correspondence assessments. It needs no Planetlib checkout to read.
There is no change to numerical code or runtime dependencies.

| Adopted model | Local correspondence | Limits |
| --- | --- | --- |
| `planetlib.solubility.sun_yao2026.binary_water@2` | `solubility.sun_yao2026.h2o_sun_yao2026`; explicit scalar EquationBinding | Distinct water partial and total melt pressures in bar; Na/Al fractions from the complete dry oxide inventory; binary H2O-equivalent mass fraction. No fugacity substitution, empirical transfer, or Gibbs continuation. |
| `planetlib.thermodynamics.nasa9.model@1` | `thermo.standard._nasa9`; one coefficient row, output tuple `(g/RT, s/R, cp/R)` | Temperature mapping is a partial EquationBinding; MCEG's scalar result mapping does not encode tuple components. No coefficient dataset, interval join, or reference-gauge adoption. |
| `planetlib.thermodynamics.ideal_mixing.model@1` | Binary restriction of `thermo.ideal_gibbs.gibbs_j` and its amount derivatives | Catalog `G` is provider `G/(RT)`; selected-pressure standards are `standard_potentials_rt + log(P/1 bar)`. Vector component selection and result normalization are documented and tested, not encoded as a misleading scalar binding. |

The ideal-mixing definition uses `G` for a quantity with units of mol;
its chemical potentials `u_i` are dimensionless. The provider returns joules.
Pressure contributions enter the supplied standards exactly once. NASA9
coefficients are ordered `(a1, ..., a7, b1, b2)` and its enthalpy is recovered
from `g/RT + s/R`. These are explicit adoption conventions, not automatic unit
or array inference.

## Maintenance and distribution

- Edit local subjects, mappings, claims and assessments in [adoption.yaml](adoption.yaml).
- Shared equations are maintained only in the upstream catalog's `Equation.sympy` fields.
- [imports.lock.json](imports.lock.json) pins full catalog files and native model dependency digests.
- [contract.yaml](contract.yaml) is a generated distribution snapshot. Do not edit its copied definitions.

MCEG schema `0.6` requires valid records and reference/pin consistency. Its
native `load_contract`, `to_sympy`, `mathematical_model_digest`,
`equation_checks` and `file_integrity` are used directly by the checks.
Planetlib's optional `tools/planetlib.py build` resolves locked definition and
provenance closures into that native schema. ExoGibbs adds no formula registry,
parser, resolver or parallel schema. The single upstream editing location and
immutable snapshot policy are project choices; MCEG does not require a Python
authoring registry or the template/snapshot layout.

The development dependency is fixed in [requirements.txt](requirements.txt).
Relevant specifications at that revision are
[symbolic source](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/symbolic.md),
[model composition](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/mathematical_models.md),
[bindings and review pins](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/equations.md), and
[assessment history](https://github.com/HajimeKawahara/mceg/blob/4034c9259c9c2e7d55243d69f3ab2f22c9cc7d88/docs/en/assessment_lifecycle.md).
Earlier optional MCEG workflows keep their existing contracts and dependencies.

Prepare an environment with this repository installed and the pinned MCEG
requirements. Obtain a separate Planetlib checkout at the commit above, then
run the following offline from the ExoGibbs root (`PLANETLIB_CHECKOUT` is its
absolute path):

```bash
python "$PLANETLIB_CHECKOUT/tools/planetlib.py" build \
  mceg/planetlib/adoption.yaml --lock mceg/planetlib/imports.lock.json \
  --output /tmp/exogibbs-planetlib-contract.yaml
diff -u mceg/planetlib/contract.yaml /tmp/exogibbs-planetlib-contract.yaml
python -m mceg mceg/planetlib/contract.yaml
python -m pytest -q tests/model_contracts/planetlib_test.py
```

The separate [adoption workflow](../../.github/workflows/planetlib.yml) acquires
the pinned development dependencies and source checkout during setup. Snapshot
reproduction and provider comparisons themselves require no network. The tests
read the distributed expressions through MCEG, compare actual provider outputs
at selected inputs, and check native model, source-file and binding pins. They
do not maintain a second set of equations.

## Review boundaries and history

All three new correspondence assessments remain `unknown`: this PR supplies
sampled regression checks, but no archived scientific review of general
provider equivalence. Planetlib's own mathematical assessments are not imported.
The original [Sun-Yao contract](../contracts/h2o_sun_yao2026.yaml), including
`numerical_verification`, `source_formula_agreement_20261021`,
`physical_accuracy` and their assessments, remains unchanged. It concerns its
own pinned provider and study conditions; the new adoption does not renew it.
NASA9 source choices and gas-mixture validation remain in their existing
[curated gas](../contracts/curated_gas.yaml) and
[gas-v2 adoption](../contracts/curated_gas_v2_adoption.yaml) records.
The shared polynomial does not endorse those datasets or transfer their support.

A catalog, convention, source or dependency change requires reviewing the
corresponding `exogibbs.planetlib.*.correspondence` claim and its pending review,
and the separately affected provider/study claims. Keep old definitions,
assessments and execution artifacts; add an explicit revision or adoption
snapshot instead of silently replacing a published definition. Updating lock
hashes alone never supplies a new judgment or execution. Physical applicability,
source correspondence, and the sampled mathematical/numerical agreement remain
distinct.

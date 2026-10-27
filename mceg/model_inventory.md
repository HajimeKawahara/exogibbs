# Model inventory: solubility and magma priority

Survey date: 2026-10-20. Audited implementation:
`7dc3ce1183c0b3d74e24cde502f638cac3625b83`.
The initial focus is the auxiliary solubility collection and its connection
conventions. Runtime equilibrium presets and source-adoption studies retain
their separate identities.

## Solubility coverage

| Contract | Driving quantity | Returned basis | Source-review boundary |
| --- | --- | --- | --- |
| Existing [H2](contracts/h2_solubility.yaml) | H2 fugacity in bar plus total melt P in GPa | H2 mole fraction | Existing primary-source review and calibration caveat preserved |
| [H2O Lichtenberg](contracts/h2o_lichtenberg2021.yaml) | H2O partial pressure in Pa | H2O mass fraction | Metadata and formula inspected; original compilation not re-inspected |
| [CO2 Lichtenberg](contracts/co2_lichtenberg2021.yaml) | CO2 partial pressure in Pa | CO2 mass fraction | Broad compilation bounds are marginal metadata |
| [CO Yoshioka](contracts/co_yoshioka2019.yaml) | CO fugacity in bar; optional natural log | Elemental-C mass fraction | Retain corrected intercept and provider-reported unit discrepancy |
| [CH4 Ardia/Seo](contracts/ch4_ardia2013.yaml) | CH4 fugacity in GPa plus total melt P in GPa | CH4 mole fraction | Separate experimental origin, formulation and later unit-label discrepancy |
| [N2 Dasgupta](contracts/n2_dasgupta2022.yaml) | N2 partial pressure in GPa, T, total melt P, delta-IW and oxide fractions | Total elemental-N mass fraction | Retain square-root total-pressure term and redox/composition dependence |
| [H2O Sun-Yao](contracts/h2o_sun_yao2026.yaml) | H2O partial pressure and total melt P in bar, T and dry Na/Al fractions | Total H2O-equivalent mass fraction | Pinned MAGEMin port; original MATLAB/experiment replay absent |

The six new contracts each include provider-reported T/P marginals and explicit
joint formula-test samples. The first five share the 56-test MELTYQ module;
Sun-Yao uses 16 tests. These are 72 executed tests in total, not six independent
experimental datasets. Physical support remains unknown. Metadata bounds are
not a validated rectangle, and sample membership does not infer interpolation.

The original Lichtenberg DOI was inaccessible (HTTP 403), and the Sun-Yao paper
and pinned online port were not retrievable during this survey. Those access
attempts did not complete source inspection. Other original solubility papers
were not re-inspected. The contracts record implementation-level findings
without claiming new agreement with the original sources.

## Other located model families

| Family | Coverage and remaining work |
| --- | --- |
| Atomic-ion reference/adoption and major-molecule adoption | Existing contracts describe source studies; they are not runtime presets |
| Gas thermochemistry impact and curated gas | Existing contracts and scoped receipts; no reinterpretation in this survey |
| FastChem/FastChem4/YKB4 and condensate presets | `src/exogibbs/presets/`; need preset-specific species/source, reference-pressure, phase and evidence scopes |
| MELTYQ magma-gas application | `src/exogibbs/applications/magma_gas/`; the complete connection needs its own basis, gas model, oxygen buffer and solver-context assessment |
| ExoEOS adapters and metal-silicate examples | `examples/`; individual solubility contracts do not validate combined gas/metal/melt equilibrium or phase selection |
| Gibbs solvers and implicit derivatives | Numerical algorithms, distinct from physical thermochemical model identity; preserve solver/environment/tolerance context separately |

## Example context

[The Sun-Yao context](contexts/sun_yao_formula_sample.json) reproduces one exact
joint formula-test input, with partial pressure and total pressure kept separate:

```bash
mceg assess mceg/contracts/h2o_sun_yao2026.yaml \
  --context mceg/contexts/sun_yao_formula_sample.json --subject survey_20261020 \
  --output /tmp/sun-yao-assessed.yaml
mceg check /tmp/sun-yao-assessed.yaml
```

This establishes only membership in reported marginals and an executed numerical
sample. `physical_review` remains unknown. Using only `melt_total_pressure`
cannot satisfy a missing `water_partial_pressure`, even when their values match.
The required oxide basis is the complete volatile-free mole inventory, and
hydroxyl is already included as H2O equivalent. The auxiliary law provides no
host chemical potentials or extensive Gibbs scalar.

## Reading and maintaining these records

The contracts use v0.2 with separate numerical and physical Assessments. The
shared [receipt](../tests/data/validation/mceg/20261020_model_survey/receipt.json),
[JUnit report](../tests/data/validation/mceg/20261020_model_survey/tests.xml) and
[execution log](../tests/data/validation/mceg/20261020_model_survey/tests.log) record fresh
CPU/x64 runs against explicit local source paths. Each contract pins its selected
source/test/fixture files and the shared artifacts. The receipt's commands and
environment describe how to repeat each module; use a new output directory.
No external reference generator, proprietary runtime, or experiment was rerun.
The selected manifest is not a complete transitive dependency lock.

Use the companion MCEG checkout containing the volatile-solubility quantity
extension (documented under Unreleased); older v0.2 readers do not recognize
all new pressure names. From this provider root with that environment active:

```bash
mceg mceg/contracts/*.yaml
mceg check . --assessment numerical_review --root .
mceg explain . --assessment physical_review
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/unittests/mceg_model_survey_test.py
```

The common numerical selector applies only to these new contracts. A numerical
PASS reports the archived assertions and matching selected files. The physical
selector intentionally remains UNKNOWN. Default `mceg check .` also shows source
review gaps and the existing historical contracts.

Local paths resolve from the provider root. A publication-only commit need not
match the audited source commit, but the selected hashes must match. After a
selected file changes, review the affected scientific declarations, run the
relevant tests, save new artifacts and add a new Subject/Context/Assessment.
Preserve earlier records; do not just replace hashes or rename an old pass.
No MCEG import or runtime dependency is added to the provider.

For a repeatable authoring procedure, see `docs/en/provider_onboarding.md` in
the companion MCEG repository. This survey's pattern is: define one model and
its conventions, inspect code and source scope, execute targeted tests, record
pins, write separate Claims/Assessments, then verify structure and local files.

## Machine-readable inventory and coverage

[models.yaml](models.yaml) is the machine-readable boundary inventory. Model
entries have code bindings, explicit choices, and zero or more contract links.
File classifications and SHA-256 pins record the reviewed inventory snapshot;
they do not renew numerical or physical evidence. Run from this provider root
with the companion MCEG environment active:

```bash
mceg coverage .
mceg coverage . --json
```

The 2026-10-20 baseline has **21 inventory entries**: 13 registered,
8 uncontracted, and 0 invalid contract mappings. These entries link all
12 provider contracts; counts differ because a family contract may serve multiple
choices. Of 209 scanned files, 21 match reviewed classifications and 188
remain unreviewed. Exit code **1** is expected. Remaining families and files are
work items, not evidence of physical failure; this is not complete model coverage.

See the [baseline report](../tests/data/validation/mceg/20261020_coverage/baseline.json) and
[mutation receipt](../tests/data/validation/mceg/20261020_coverage/mutations.json). Eight mutations
on a temporary copy produced the expected new diagnostics: a missing model link,
a missing contract, a wrong contract ID, a new formula branch, a renamed function,
a new source file, changed coefficient data, and an unlisted contract. A ninth
case records the detection limit: deleting a whole variant row while other rows
still map its file and family contract introduces no diagnostic. Initial model
classification and inventory diffs therefore require human review.

Source scans include all packaged files, not just public functions or contract
manifests. Known uncontracted families remain explicit; unfamiliar files remain
unreviewed. Review any code/data change before refreshing the inventory digest,
and update affected contracts/evidence separately. The companion MCEG guide
`docs/en/coverage.md` specifies the format, diagnostics and remaining limits.

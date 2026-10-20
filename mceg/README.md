# Experimental MCEG contracts

[MCEG](https://github.com/HajimeKawahara/mceg) records model assumptions,
applicability, claims and scoped evidence. This checkout pilots its frozen
`schema_version: "0.1"` with the
[H2 solubility law](contracts/h2_solubility.yaml). MCEG is not an ExoGibbs
runtime or test dependency; numerical implementations remain owned by ExoGibbs.

The [2026-10-20 model inventory](model_inventory.md) adds v0.2 contracts for the
solubility/magma survey, with separate numerical and physical Assessments and
fresh receipts. The new records use the companion MCEG volatile-quantity
extension. Existing H2 and source-study histories retain their original scope.

The [atomic-ion reference contract](contracts/atomic_ion_reference.yaml) records
the separate gas-only audit of 49 FastChem4 atomic ions. Its
[report](../documents/atomic_ion_audit.rst),
[numerical receipt](../results/atomic_ion_audit/receipt.json) and
[test receipt](../results/atomic_ion_audit/tests.json) describe the pinned sources,
47 numerical references and two anions without a stable bound-state reference.
The audit uses the standard library; the existing MCEG v0.1 schema needs no
extension. The sections below describe the original H2 pilot workflow.

The [atomic-ion adoption contract](contracts/atomic_ion_adoption.yaml) records
the subsequent [49-species source choices](../documents/atomic_ion_adoption.rst):
43 NASA9 thermal models with spectroscopic threshold updates, four partition
models, and two exclusions. The [new receipt](../results/atomic_ion_adoption/receipt.json)
and [test record](../results/atomic_ion_adoption/tests.json) keep this study
separate from the original audit. The CDS original has now been compared with
the local partition copy. The choices are not a production preset; continuous
partition representations and application-specific validation remain necessary.

The [major-molecule adoption contract](contracts/major_molecule_adoption.yaml)
records [35 neutral-molecule source decisions](../documents/major_molecule_adoption.rst)
following approval of the atomic policy. It distinguishes energy anchors from
thermal functions and retains unresolved source choices explicitly. The
[adoption table](../results/major_molecule_adoption/adoption_table.csv),
[numerical receipt](../results/major_molecule_adoption/receipt.json) and
[test record](../results/major_molecule_adoption/tests.json) are offline study
artifacts. Neither this study nor its contract installs a runtime preset.

The [gas-composition impact contract](contracts/gas_thermochemistry_impact.yaml)
records the subsequent [fixed-network comparison](../documents/gas_thermochemistry_impact.rst)
of the approved atomic and molecular choices. Its
[receipt](../results/gas_thermochemistry_impact/receipt.json) distinguishes
numerical verification, optional independent FastChem execution, and the limited
physical evidence. All five models share the same 537 species and elemental
budget. The [standalone FastChem run](../results/gas_thermochemistry_impact/fastchem_parity.csv)
now verifies all 100 cases using the
[pinned v4.0.3 build](../results/gas_thermochemistry_impact/fastchem_build.json).
This is implementation evidence; the complete hybrid's physical accuracy
remains unestablished.

The [curated gas contract](contracts/curated_gas.yaml) records the subsequent
[continuous runtime preset](../documents/presets/curated.rst),
`curated-gas-v1`, on 1500--4000 K. Its
[validation receipt](../results/curated_gas_validation/receipt.json) covers
the 537-species ion-inclusive and 421-species neutral networks, continuous
source representations and independent equilibrium comparisons. Runtime
code consumes pinned numerical JSON only; MCEG remains an optional development
tool. Earlier source-study contracts and their archived results are historical
evidence and are not rewritten by this implementation.

The curated receipt has a separate offline consistency check:

```bash
PYTHONPATH=src python -m pytest -q tests/unittests/examples/curated_receipt_test.py
```

It checks the complete recorded input/output hash lists, the 70 completed
FastChem comparisons, and the contract's implementation hash. The FastChem
binary and MCEG are unnecessary for this check. A changed input requires review
and a new numerical receipt; updating hashes alone does not establish a new
passing run. Earlier test receipts remain historical execution records.
These checks establish recorded-file consistency, not physical accuracy or
coverage of new compositions and conditions.

## Ownership and references

The [2026-10-19 Hirschmann literature review](reviews/h2_literature_update/README.md)
records inspection of the original published regression and material-specific
run bounds. Its [offline replay result](../results/mceg/h2_literature_update.json)
preserves earlier evidence and unknown physical support. The archived baseline
and authored review live here; MCEG provides optional generic tooling only.
Replay requires MCEG `0.2.0rc2` or later and does not require the private PDF,
ExoEOS, or an installed ExoGibbs package. This review retains its historical
implementation identity and does not replace the current numerical receipt.

ExoGibbs owns the solubility contract and ExoEOS owns the pure-H2 EOS contract.
The fixed 1700 K, 7500 bar connection, manual applicability assessments and
combined execution record remain a consumer example in MCEG. Neither provider's
contract alone establishes the physical accuracy of the combined calculation.

Local `sources` paths are relative to the **ExoGibbs repository root**, not this
directory, the YAML's directory or the caller's working directory. Current code
references are local. Historical tests and the original MCEG audit retain
immutable commit URLs. The partial primary-source inspection and differing
source-specific temperature intervals are preserved without new scientific claims.

The implementation revision is
`c2d3e8b0490d4ae502720940ea220a21a3146a56`. Sun--Yao water adds public exports and changes the
export-list test; the H2 scalar, metadata, fugacity conversion and adapter
remain byte-identical to the previous H2 receipt. The eight scoped tests were
freshly rerun with ExoEOS `9dc71506264b220fbd496259060cb67a95c1d11c`. No H2 scientific declaration,
applicability assessment or consumer case input changed.

[The current receipt](../results/mceg/h2_tests_c2d3e8b.json) and
[JUnit report](../results/mceg/h2_tests_c2d3e8b.xml) preserve eight passing tests,
the actual environment/imports and selected source hashes checked against Git
objects. The [previous receipt](../results/mceg/h2_tests_748f9da.json) and older
records retain their original scope. This refresh validates the unchanged H2
connection; it supplies no empirical evidence for the new Sun--Yao water law.
The implementation revision is distinct from the commit publishing the contract.

## Validation and review

Use an independent MCEG environment for optional structural validation:

```bash
PYTHONPATH=/path/to/mceg/src python -m mceg \
  /path/to/exogibbs/mceg/contracts/h2_solubility.yaml
```

Structure does not establish reference availability, source inspection,
numerical correctness or scientific applicability. The ordinary provider suite
checks the recorded local file hashes and the contract/guide references without
importing MCEG. The reference check targets this pilot's fixed text layout;
it does not parse or validate general YAML:

```bash
PYTHONPATH=src python -m pytest -q tests/unittests/mceg_contract_test.py
```

The receipt records the exact numerical test selection. Its optional ExoEOS
connection check needs the audited ExoEOS checkout on PYTHONPATH. No ExoEOS or
MCEG dependency is introduced by the metadata check itself.

A commit adding only contracts can change HEAD without changing the audited
files. Compare their hashes; do not require a contract to contain its own future
commit hash. Selected-file changes require manual review of affected claims,
source comparisons and scientific scope. The check is conservative and covers
only the recorded files; it does not infer scientific impact or complete
transitive dependencies. Rerun affected tests and create a new receipt, keeping
historical results attached to their original revisions. An unexecuted current
test is `not_run`, not a new pass or an observed failure.

## Updating the evidence

1. Compare the selected files with the previous audited revision and review
   affected claims. A change elsewhere in the adapter still changes its whole-file
   hash; numerical reruns and scientific review have separate scopes.
2. Run the receipt's numerical test selectors with explicit provider checkouts.
   Save a new JSON receipt and JUnit report, including the actual revisions,
   dependency versions, import locations and selected-file hashes. Verify the
   hashes against the recorded Git objects and retain previous artifacts.
3. Update the contract's implementation revision, `provider_test_run` sources
   and scope, this guide, and `RECEIPT` in the metadata test together. Updating
   only the test's receipt leaves consumers reading historical evidence.
4. Run the metadata tests, independent MCEG structural validation and the
   consumer replay below. Changed scientific declarations or case inputs require
   a new manual assessment before reusing the consumer's classifications.

```bash
PYTHONPATH=/path/to/mceg/src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
python /path/to/mceg/examples/provider_h2/run.py \
  --exoeos /path/to/exoeos --exogibbs /path/to/exogibbs \
  --output /tmp/provider-h2-result.json
```

This workflow can reveal stale evidence and reproduce scoped numerical checks.
It does not establish physical accuracy, complete dependency coverage, or an
automatic interpretation of applicability. The model's physical-validation
claim remains `unknown`.

The JSON receipt is a pilot artifact, not a v0.1 schema extension. These files
are checkout metadata; wheel-level discovery and automated scientific
applicability decisions remain outside this pilot.

## Inventory coverage

The [machine-readable inventory](models.yaml) maps code and model choices to
contracts, including known uncontracted families. `mceg coverage .` checks these
links and reports new/changed or unreviewed source files without importing this
package. The current partial inventory intentionally returns exit code 1.
See the [inventory notes](model_inventory.md#machine-readable-inventory-and-coverage)
for baseline counts, mutation checks and the limits of implicit-model detection.

## Models and evidence HTML

Run `python -m mceg docs --config mceg/docs.yaml` to generate RST only, or
`./update_doc.sh` from the repository root to regenerate and build HTML.
See [the documentation build guide](../documents/mceg/build.rst) for installation,
input selection and output locations. Generated files are ignored by Git.

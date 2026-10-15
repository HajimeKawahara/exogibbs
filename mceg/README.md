# Experimental MCEG contracts

[MCEG](https://github.com/HajimeKawahara/mceg) records model assumptions,
applicability, claims and scoped evidence. This checkout pilots its frozen
`schema_version: "0.1"` with the
[H2 solubility law](contracts/h2_solubility.yaml). MCEG is not an ExoGibbs
runtime or test dependency; the scientific implementation is unchanged.

## Ownership and references

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
`9990fdc2f4bd8423bdfc0cefdc1fbb8325cfe55f`, the develop snapshot reviewed before
this metadata-only addition. The solubility source is unchanged from the earlier
audit. The updated adapter adds multiphase functions; its existing pure-H2
callback and unit conversion are unchanged. The scoped tests were rerun.

[The new receipt](../results/mceg/h2_tests_9990fdc.json) and
[JUnit report](../results/mceg/h2_tests_9990fdc.xml) preserve eight passing tests,
the environment, selected source hashes and the ExoEOS dependency used in the
connection test. The old run remains historical evidence; the current numerical
claim links to `provider_test_run`. The implementation revision is distinct
from the commit publishing the contract.

## Validation and review

Use an independent MCEG environment for optional structural validation:

```bash
PYTHONPATH=/path/to/mceg/src python -m mceg \
  /path/to/exogibbs/mceg/contracts/h2_solubility.yaml
```

Structure does not establish reference availability, source inspection,
numerical correctness or scientific applicability. The ordinary provider suite
checks the recorded local file hashes without importing MCEG:

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

The JSON receipt is a pilot artifact, not a v0.1 schema extension. These files
are checkout metadata; wheel-level discovery and automated scientific
applicability decisions remain outside this pilot.

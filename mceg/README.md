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
`748f9da6a7f13e9082fa42ab61caca907c39fe5b`, the develop snapshot reviewed before
this maintenance update. Since the original audit, the adapter gained a magma
derivative-domain callback. Its pure-H2 callback and scalar input conversion
are unchanged; the solubility and unit-conversion files are byte-identical.
The scoped tests were rerun with ExoEOS main at
`50f3a20ecca5fdc8b422a8869ccf2e5d02a8fc01`.

[The current receipt](../results/mceg/h2_tests_748f9da.json) and
[JUnit report](../results/mceg/h2_tests_748f9da.xml) preserve eight passing tests,
the environment, selected source hashes and the ExoEOS dependency used in the
connection test. Both providers' selected files were checked against Git objects
at their recorded revisions. The [adoption run](../results/mceg/h2_tests_9990fdc.json)
and [intermediate run](../results/mceg/h2_tests_47ecf87_r5.json) remain historical
evidence; the current numerical claim links to `provider_test_run`. The
implementation revision is distinct from the commit publishing the contract.

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

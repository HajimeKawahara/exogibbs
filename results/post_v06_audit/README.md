# Post-v0.6 audit

## Scope

Reviewed the changes from `v0.6.0` to develop
`7fdbb700f5a567ce9535548da98fa6da5dc78614` (99 commits including merges,
264 changed files). The audit focused on production numerical paths, the
ExoInventory-facing interfaces, their examples, regression coverage, and CI.
The existing uncommitted documentation/example runners and Japanese
documentation submodule checkout were excluded from this PR.

## Findings and changes

| Area | Finding | Change |
| --- | --- | --- |
| Condensate initialization | Rainout scaling and canonical normalization duplicated the same field transformations; only rainout scaled the inventory-bridge origin. | Share the host amount-gauge transforms and scale the origin with the initial state. Preserve direct division and subnormal amounts. |
| Rainout inventory | Certification and propagation separately implemented the same depleted-species mask and gas/condensate inventory contractions. | Share the contractions while retaining separate certification and propagation decisions, raw diagnostics, and charge handling. |
| Gas profiles | Diagnostic and ordinary paths repeated both scan and vmap scheduling. | Use the static diagnostics flag in one path for each scheduling method. Preserve explicit seeds, warm continuation, result order, and fugacity callbacks. |
| FastChem presets | Both condensate presets duplicated the missing-element check. | Share the check over ordered entries, preserving every duplicate species slot. |
| ExoEOS adapters | T/P conversion was duplicated. The all-ideal pure adapter silently ignored non-scalar T/P, whereas the solution adapter rejected them. | Share scalar validation, dtype promotion, and the existing bar-to-Pa conversion; require `jax.vmap` for batches. |
| Source activities | Young and sulfur/carbon source models duplicated the GCE Si/O coefficient equations. | Share only those equations; retain full-metal mole fractions, network-specific corrections, and the independent reference calculations. Include the shared source in M1 provenance hashes. |
| Provider CI | Temporary stacked-PR logic silently omitted three integration files if absent. | Require those now-existing files explicitly. |

Regression coverage includes gauge round trips down to `1e-310`, bridge-origin
inventories, depleted species and charge rows, all three gas profile schedules
with and without diagnostics, every duplicate condensate slot, scalar provider
inputs under JIT, and the shared activity source's provenance.

## Intentionally retained

- The zero-barrier rank-deficiency, trace-inventory, structural-zero, and
  optimizer-limit routes have distinct acceptance contracts and regression
  cases. They were not replaced with one generic retry or split mechanically.
- The experimental magma-gas and MELTYQ import paths are compatibility facades
  over the existing implementation, not duplicate solver implementations.
- The independent NumPy/reference activity equations and frozen archives remain
  independent audit evidence. Source-network physics, empirical corrections,
  phase assumptions, and approximate M1 acceptance limits were not changed.

## Validation

Validation uses offline local inputs, Python 3.9.19, JAX 0.4.30, NumPy 1.26.4,
SciPy 1.13.1, CPU float64, and one BLAS/OpenMP thread. Commands are run from the
audit worktree with `PYTHONPATH=src:.:/home/kawahara/exoeos/src`.

- Full suite: `python -m pytest tests/unittests --junitxml=results/post_v06_audit/pytest.xml`.
- Pinned provider: 92 passed in 40.18s, zero skips. ExoEOS
  `0c85dfe28353bf70d7d687e49689e74db556c4b0` was extracted from local git objects
  and imported from `/tmp/exogibbs-audit-provider-pin/src`; the exact CI provider
  suite was run with this source first on `PYTHONPATH`.
- Documentation: `./update_doc.sh` completed with 112 warnings. A clean build
  of the starting develop commit produced the same warning multiset after
  normalizing paths and line numbers; this change adds no warnings.
- `git diff --check` passed.
- The user's local comprehensive runner dry-run lists 46 jobs and three
  compatibility aliases. Full GPU examples/retrievals are the user's follow-up
  run; they were not run or claimed as validated by this CPU audit.

Full-suite output:

```text
1575 passed, 1 skipped, 24 warnings in 1729.49s (0:28:49)
```

The only skip is `test_package_version_matches_scm_generated_module`: this
worktree has no setuptools-scm-generated version file. Installed metadata
fallback is covered by the adjacent passing test. The full-suite, pinned
provider, and documentation logs and JUnit reports remain local artifacts in
this directory.

# Equilibrium architecture refactor handoff

## 1. Purpose

This document is the handoff for the refactor that follows the fixed-support
v1 runtime retirement merged in PR #140.

The next refactor should make the two user-facing equilibrium modes easier to
discover and develop:

1. gas-only chemical equilibrium;
2. gas-plus-condensate equilibrium.

These modes should have symmetric public entry points. Their internal
implementations should not be forced into an artificial mirror, because the
gas-plus-condensate lifecycle intentionally uses the gas-only solver for
initialization, activity-driven support selection, and valid gas-only terminal
outcomes.

The intended dependency is:

```text
gas-only API
    -> gas solver

gas-plus-condensate API
    -> gas solver
    -> condensate support lifecycle
    -> fixed-support condensate solver
```

## 2. Repository checkpoint

As of 2026-07-27:

```text
workspace: /home/kawahara/exogibbs-condensate-cleanup
branch:    refactor2
HEAD:      40e2024 Merge pull request #140 from HajimeKawahara/cleanup/condensate-legacy
base:      origin/develop at 40e2024
status:    tracked worktree clean
```

The original `/home/kawahara/exogibbs` worktree must not be used or modified.

The system `PYTHONPATH` may still contain the original workspace. Use:

```bash
cd /home/kawahara/exogibbs-condensate-cleanup
env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -c "import exogibbs; print(exogibbs.__file__)"
```

Expected:

```text
/home/kawahara/exogibbs-condensate-cleanup/src/exogibbs/__init__.py
```

`refactor2` currently has no upstream branch. Do not push it over
`origin/develop`; publish it as `origin/refactor2` when a reviewed checkpoint
is ready.

## 3. Production contract that must remain unchanged

The runtime condensate route is v2-only:

- route: `head_v2`;
- route version: `v2.0`;
- production preset: `validated_2026_07`;
- support lifecycle is outside the fixed-support solver;
- no automatic v1 fallback exists;
- solver failure propagates without selecting a retired solver;
- rollback is performed at release-artifact level.

The executable v1 implementation must not be restored as an import facade,
fallback, test helper, or benchmark dependency.

Frozen v1 JSON/Markdown evidence and its SHA-256 verification remain historical
evidence. Promotion-era validation and design documents remain historical
records even when their internal descriptions are no longer current runtime
instructions.

Authoritative production documents:

1. `documents/fixed_support_solver_v2_production_migration.md`
2. `documents/fixed_support_solver_v2_validation.md`
3. `documents/fixed_support_solver_v2_design.md`
4. `documents/condensate_profile.rst`
5. `benchmarks/fixed_support_v2/README.md`
6. `refactory_strategy.md`

## 4. Validation checkpoint inherited from PR #140

The merged source was validated with:

```text
compileall src                         passed
full unit tests                       215 passed, 1 warning
frozen v1 artifact hashes             2/2
A100 production preflight             10/10
A100 cold/warm families               10/10
A100 cold/warm layers                 108/108
exclusive GPU measurement             true
runtime budget checks                 4/4
production_profile_gate_passed        true
promotion_authorized                  true
```

The A100 artifact was measured from source commit
`d769da4662ad0b7eefc4b8ae21f6398baf583f18`, which is the source parent of the
merge checkpoint.

Artifact SHA-256:

```text
production_preflight.json  47640313b4f4aef8b033dd8fada8fd5b51bcdd9e004a12354ee2ef89d4131cf5
summary.json               e66a81e4d81df442b603bce1ce4968b8fd573267d3a2fe0189b22696e6de9864
summary.md                 7f0fe74fd5754ac39abfc54a5e15122247e0d402e893d113db5a734f1b2e485d
```

The local artifacts are under
`results/fixed_support_v2_production_profile/` and are ignored. They should not
be committed merely to preserve the refactor record.

## 5. Current structure and confirmed asymmetries

### 5.1 User-facing API

The direct module entry points are reasonably symmetric:

```text
exogibbs.api.equilibrium
    equilibrium
    equilibrium_profile

exogibbs.api.condensate_equilibrium
    condensate_equilibrium
    condensate_equilibrium_profile
```

The umbrella package `exogibbs.api` is not symmetric or stable:

- `api.equilibrium` resolves as a module;
- a first direct attribute read of `api.condensate_equilibrium` can return the
  function, while the import side effect installs the submodule and a later
  read returns the module;
- `from exogibbs.api import condensate_equilibrium` resolves to the submodule,
  despite the lazy-export branch intending to return the function;
- `equilibrium_profile` is absent from `exogibbs.api.__all__`;
- gas and condensate Init, Initializer, ProfileResult, and request types are
  exposed inconsistently.

Do not change this surface without first adding import-contract tests and
choosing an explicit compatibility policy.

### 5.2 Runtime ownership

Gas-only runtime is stored under generic optimizer names:

```text
src/exogibbs/api/equilibrium.py                 531 lines
src/exogibbs/api/equilibrium_grid.py          1050 lines
src/exogibbs/optimize/minimize.py             1438 lines
src/exogibbs/optimize/core.py
src/exogibbs/optimize/vjpgibbs.py
```

Gas-plus-condensate runtime is spread across three ownership locations:

```text
src/exogibbs/api/condensate_equilibrium.py     2328 lines
src/exogibbs/condensates/
src/exogibbs/optimize/fixed_support_v2_profile.py
src/exogibbs/optimize/fixed_support_v2/
```

`condensate_equilibrium.py` imports and calls the gas-only public solver for:

- gas-only full-budget activity state;
- v2 initial gas state;
- lifecycle startup;
- layers with no eligible condensate candidate.

This dependency is intentional. The directory structure should make it
visible rather than pretending the solvers are independent peers.

### 5.3 Naming

The current names do not consistently describe ownership:

- `optimize/minimize.py` is specifically the gas-only Gibbs solver;
- `fixed_support_v2` looks experimental/versioned even though it is now the
  only production fixed-support implementation;
- `condensates`, `condensate_equilibrium`, and preset suffix `_cond` use three
  naming conventions for the same domain;
- `fixed_support_v2_profile.py` is a sibling of the package whose internals it
  orchestrates.

Keep public route metadata at `v2.0`; renaming an internal package does not
authorize changing the production route version.

### 5.4 Tests, benchmarks, examples, and documentation

Tests do not mirror the source ownership:

- gas optimizer tests are under `tests/unittests/optimize/lagrange/`;
- fixed-support tests are flat `fixed_support_v2_*` files under
  `tests/unittests/optimize/`;
- condensate lifecycle tests are split between `api/`, `condensates/`, and
  `benchmarks/`.

Gas benchmarks are at `benchmarks/` root, while the condensate production gate
is under `benchmarks/fixed_support_v2/`. Examples are now mostly condensate
examples, while the root README is gas-only.

Active and historical documents are mixed at `documents/` root. At least these
historical documents contain paths or claims for deleted v1 modules and must
not be treated as current runtime instructions:

- `documents/audit_condensates.md`
- `documents/ipm_audit.md`
- `documents/ipopt_exogibbs_mathnote.md`
- `documents/pdipm_math_contract_audit.md`

`documents/index.rst` references the absent `documents/exogibbs/index.rst`.
The repository guidelines mention `./update_doc.sh`, but that script is absent
at this checkpoint. Documentation reorganization must repair or deliberately
replace this generation contract.

## 6. Confirmed low-risk cleanup candidates

These are candidates, not instructions to delete without a local recheck.

### 6.1 `optimize/stepsize.py`

Repository-wide search found no imports or call sites for:

- `stepsize_cea_gas`
- `stepsize_cond_heurstic`
- `stepsize_sk`

The latter two are retired condensate heuristics. The file is not exported by
`exogibbs.optimize`. Reconfirm with `rg` before deletion and add the result to
the strategy log.

### 6.2 `src/exogibbs/test/`

The package contains analytic systems and reference helpers used only by unit
tests. `generate_gibbs.py` also contains a developer-specific absolute input
path. Audit whether any of these names are intentionally public; otherwise
move the fixtures to `tests/` and remove `exogibbs.test` from packaged runtime
and generated API documentation.

### 6.3 Gas solver diagnostic cluster

`optimize/minimize.py` still begins with a large source-trace/long-double
diagnostic cluster. The following symbols had no repository call sites outside
that cluster at handoff time:

- `minimize_gibbs_core_with_source_trace`
- `profile_minimize_gibbs_iterations`
- `build_minimize_gibbs_core_lnnk_output_source_trace`
- `build_minimize_gibbs_core_final_carry_source_trace`

Audit direct-public compatibility and all tests/docs before deletion. The live
`minimize_gibbs`, `minimize_gibbs_with_diagnostics`, custom VJP, and ordinary
convergence diagnostics must remain.

### 6.4 Curated production cases

`exogibbs.condensates.curated_profiles` is consumed by the benchmark gate,
examples, and tests, but not by the production API. Decide whether immutable
gate cases belong under `benchmarks/` rather than the installed runtime
package. Do not move them until example execution and package-data behavior
are understood.

### 6.5 Placeholder API

`LearnedEquilibriumInitializer` only raises `NotImplementedError` and is tested
only as a placeholder. Decide whether it belongs in the public API or should
be removed until an implementation exists.

## 7. Recommended target structure

Use stable API facades and feature-oriented internal ownership:

```text
src/exogibbs/
├── api/
│   ├── gas.py
│   ├── condensate.py
│   ├── equilibrium.py                 # compatibility facade
│   ├── condensate_equilibrium.py      # compatibility facade
│   └── chemistry.py
│
├── equilibrium/
│   ├── common/
│   ├── gas/
│   │   ├── types.py
│   │   ├── solver.py
│   │   ├── profile.py
│   │   ├── diagnostics.py
│   │   └── autodiff.py
│   │
│   └── condensate/
│       ├── types.py
│       ├── lifecycle.py
│       ├── support.py
│       ├── profile.py
│       ├── result.py
│       └── fixed_support/
│           ├── controller.py
│           ├── problem.py
│           ├── normal.py
│           ├── restoration.py
│           ├── continuation.py
│           └── types.py
```

This tree is a direction, not a requirement to perform one large move.
Prefer small commits that preserve symbol identity and behavior.

A possible future symmetric API is:

```python
from exogibbs.api.gas import solve, solve_profile
from exogibbs.api.condensate import solve, solve_profile
```

Do not introduce these names until naming, return types, and compatibility
facades are explicitly decided.

## 8. Recommended implementation waves

### Wave 0: guardrails and decision record

1. Reconfirm current imports and call sites with `rg`.
2. Add tests for direct module imports and `exogibbs.api` attribute/from-import
   behavior.
3. Record the chosen canonical user imports and compatibility lifetime.
4. Define ownership rules for `api`, `equilibrium`, solver, benchmark, and
   historical-document modules.

No solver code should move in the same commit as the public API decision.

### Wave 1: zero-reachability and documentation cleanup

1. Delete `optimize/stepsize.py` after rechecking zero call sites.
2. Move test-only helpers out of `src/exogibbs/test/` if the public audit
   confirms they are not supported API.
3. Move historical audit notes under `documents/history/condensates/`, add
   prominent historical banners, and update all references/toctrees.
4. Repair the Sphinx API index/update contract.

Run full unit tests because fixture imports and documentation provenance are
package-wide concerns.

### Wave 2: public facade normalization

1. Introduce unambiguous gas and condensate API modules.
2. Keep `exogibbs.api.equilibrium` and
   `exogibbs.api.condensate_equilibrium` as compatibility facades.
3. Make `__all__` deterministic; do not return a function on first access and
   a module on later access.
4. Document one-layer, profile, setup, init, result, and diagnostics symmetry.

Treat any removal from a documented import path as a deliberate compatibility
decision.

### Wave 3: gas implementation extraction

1. Separate live solver, custom VJP, diagnostics, and profile orchestration.
2. Remove the dead source-trace cluster only after its own reachability audit.
3. Rename generic internal files to gas-owned names.
4. Keep compatibility imports long enough to avoid mixing a move with a public
   removal.

### Wave 4: condensate implementation extraction

1. Reduce `api/condensate_equilibrium.py` to public types, validation, and
   facade calls.
2. Extract result construction and the full-budget gate.
3. Extract support discovery/expansion and outer lifecycle orchestration.
4. Move `fixed_support_v2_profile.py` beside the fixed-support implementation.
5. Consider dropping `_v2` from internal package names only after import and
   benchmark references have migrated.

The fixed-support solver must continue to own exactly one fixed-support solve;
support discovery and closure remain outside it.

### Wave 5: mirror repository support surfaces

Organize tests, benchmarks, examples, and active documentation by:

```text
gas/
condensate/
common/
history/
```

Do not move frozen artifact paths without updating and verifying recorded
SHA-256 declarations.

## 9. Validation commands

Always pin the new worktree source:

```bash
env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q tests/unittests
```

Useful targeted groups:

```bash
env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q \
  tests/unittests/api/equilibrium_api_test.py \
  tests/unittests/api/equilibrium_profile_test.py

env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q \
  tests/unittests/api/condensate_equilibrium_test.py \
  tests/unittests/api/condensate_equilibrium_profile_test.py \
  tests/unittests/api/condensate_production_route_contract_test.py

env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q \
  tests/unittests/optimize/fixed_support_v2_controller_test.py \
  tests/unittests/optimize/fixed_support_v2_problem_test.py \
  tests/unittests/optimize/fixed_support_v2_restoration_test.py \
  tests/unittests/optimize/fixed_support_v2_profile_test.py
```

Before each commit:

```bash
git diff --check
git status --short --branch
```

Run the A100 production-profile gate after changes that affect runtime imports,
JAX callable identity, profile orchestration, fixed-support implementation, or
production policy:

```csh
cd /home/kawahara/exogibbs-condensate-cleanup
source .venv/bin/activate.csh
setenv PYTHONPATH /home/kawahara/exogibbs-condensate-cleanup/src
csh benchmarks/fixed_support_v2/run_fixed_support_v2_production_profile_gpu.csh
```

Pure documentation moves do not require an A100 rerun. Pure Python module moves
may still affect JAX tracing/caching and should receive the gate at the end of
their wave.

## 10. Rules for the next session

- Conversation and status reports are in Japanese.
- Code, comments, and docstrings are in English.
- Use `rg` or `rg --files` for reachability work.
- Inspect large files by symbol and call site rather than reading them
  unconditionally.
- Preserve unrelated user changes.
- Do not copy ignored artifacts from `/home/kawahara/exogibbs`.
- Do not restore v1 executable paths for convenience.
- Do not perform a large directory move and behavior change in one commit.
- Update this handoff or a successor strategy record after every completed
  wave.

## 11. Suggested first prompt

```text
documents/equilibrium_architecture_refactor_handoff.mdをauthoritative handoffとして読み、
Wave 0から開始してください。まずrefactor2とorigin/developの一致、worktree、
PYTHONPATH/import provenanceを確認し、public import contractとzero-reachability候補を
再監査してください。最初から大規模moveや削除はせず、監査結果と最初のsmall commit
候補を提示してください。会話・報告は日本語、code/comments/docstringsはEnglish。
```

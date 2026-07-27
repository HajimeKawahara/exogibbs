# Equilibrium solver architecture refactoring strategy

## 0. Status and scope

| Item | Value |
|---|---|
| Status | Working architecture decision; implementation has not started |
| Audit date | 2026-07-27 |
| Audited branch | `refactor2` |
| Audited source | `fab1c6b40713` |
| Primary scope | Gas-only and gas-plus-condensate equilibrium architecture |
| Compatibility scope | Existing documented `exogibbs.api.*` imports |

This document records the architecture chosen after inspecting the current
source, tests, examples, and documentation. The earlier
`documents/equilibrium_architecture_refactor_handoff.md` was used as evidence,
not as an authoritative plan. Where this document is more specific, this
document is the working strategy for the next refactor.

The central decision is:

> Present gas-only and gas-plus-condensate equilibrium as symmetric user
> capabilities, but preserve their intentional internal asymmetry. The
> condensate capability depends on the gas solver and a fixed-support
> condensate kernel; the gas capability must not depend on condensate code.

This is an architecture refactor, not permission to change numerical behavior,
production policy, convergence criteria, or scientific results while moving
code.

## 1. Product model and terminology

There are two user-facing equilibrium capabilities but three distinct runtime
responsibilities.

| Term in this document | Meaning | Current public entry point |
|---|---|---|
| **gas equilibrium** | Equilibrium with gas species only | `equilibrium`, `equilibrium_profile` |
| **condensate equilibrium** | Equilibrium of gas plus zero or more condensed phases | `condensate_equilibrium`, `condensate_equilibrium_profile` |
| **fixed-support solve** | One inner numerical solve for a caller-supplied set of positive condensates | Not a peer user-facing solver |
| **support lifecycle** | Discover, seed, solve, check closure, and monotonically expand condensate support | Embedded in the current condensate API module |

The fixed-support solver is not a third product mode. It is an implementation
kernel used by condensate equilibrium. Likewise, a valid condensate-equilibrium
outcome may contain no condensate and may be obtained from the gas solver.

The current production support lifecycle only adds phases. It does not remove
or prune them. Until removal exists, use **monotone support discovery** or
**support lifecycle**, not **active-set method**.

## 2. Current architecture audit

### 2.1 Runtime paths

The gas path is:

```text
presets/* -> ChemicalSetup
          -> api/equilibrium.py
          -> optimize/minimize.py
          -> optimize/core.py + optimize/vjpgibbs.py
```

The one-layer API constructs a `ThermoState`, calls `minimize_gibbs`, and
constructs an `EquilibriumResult`. The profile API schedules the same solve with
`jax.vmap` or `jax.lax.scan`.

The condensate path is:

```text
presets/*_cond
  -> CondensateChemicalSetup
  -> api/condensate_equilibrium.py
     -> gas equilibrium for activity and initialization
     -> condensates/* for support selection and seeding
     -> optimize/fixed_support_v2_profile.py
        -> optimize/fixed_support_v2/*
     -> inactive-condensate closure
     -> support expansion and repeated fixed-support solves
     -> result post-processing and budget acceptance
```

The numerical core inside `optimize/fixed_support_v2/` already has a mostly
coherent internal dependency order:

```text
types
  -> problem, filter, linear_solver
  -> normal, soc, restoration, return_map
  -> controller
  -> continuation
```

The main architectural problem is therefore not the fixed-support mathematics.
It is the ownership and connection of the public API, gas foundation, host-side
support lifecycle, and fixed-support kernel.

### 2.2 Main structural problems

1. **Gas ownership is implicit.**

   Gas-only implementation uses generic names such as
   `api/equilibrium.py`, `optimize/minimize.py`, and `optimize/core.py`.
   Condensate code alone is feature-named. The source tree therefore hides the
   two user capabilities and makes gas code look universally shared.

2. **API modules own production implementation.**

   `api/equilibrium.py` is 531 lines and owns types, initialization, profile
   scheduling, caching, validation, and result construction.
   `api/condensate_equilibrium.py` is 2,328 lines and additionally owns support
   discovery, gas initialization, bucket preparation, lifecycle rounds,
   acceptance, result post-processing, and experimental prepared plans.
   `api/` is not a facade layer today.

3. **Dependency direction is inverted around shared models.**

   The gas optimizer imports `ThermoState` from `api.chemistry`; presets import
   `ChemicalSetup` from the same API module. The FastChem condensate preset also
   imports its setup builder from `api.condensate_equilibrium`. Domain and
   numerical code therefore depend upward on the public API.

4. **The umbrella API is import-order dependent.**

   `exogibbs.api` uses lazy attributes whose names collide with child modules.
   In a fresh process, `api.condensate_equilibrium` can first be a function and
   later become a module after the submodule import. A star or from-import
   installs the module. `api.equilibrium` is a module, while
   `equilibrium_profile` is absent from the umbrella `__all__`.

   A package attribute must not be both a callable and a child-module name.

5. **Numerical and diagnostic gas code is mixed.**

   `optimize/minimize.py` is 1,438 lines. The live CEA-style Newton iteration,
   custom VJP, ordinary diagnostics, profiling, long-double comparison, and
   source-trace tooling share one module. This obscures the small production
   kernel and makes reachability hard to establish.

6. **Condensate execution boundaries are hidden.**

   The public condensate module mixes JAX kernels with Python control flow,
   NumPy, SciPy least squares, `jax.device_get`, weak-reference caches, and wall
   clock measurements. The public API is host-orchestrated even though its
   fixed-support inner solves are compiled JAX computations.

7. **Tests mirror historical locations rather than ownership.**

   Gas-kernel tests live under `tests/unittests/optimize/lagrange/`, although no
   corresponding source package exists. Fixed-support tests are flat
   `fixed_support_v2_*` files. Lifecycle tests live under `api/` and mostly
   monkeypatch the actual solver. A small real-preset, public-API,
   real-fixed-support end-to-end test is missing from the unit suite.

### 2.3 Boundary defects exposed by the audit

These findings need dedicated characterization or correctness changes. They
must not be silently altered by file moves.

1. **Optional metadata is used as runtime numerical input.**

   `ChemicalSetup.metadata` is optional and described as provenance, but
   condensate support selection reads `temperature_validity_upper` from it.
   The annotation says `Mapping[str, str]` although the preset stores a tuple of
   floats. A setup with `metadata=None` can fail on `.get`.

2. **The result builder can change the numerical solution.**

   `build_condensate_equilibrium_result_from_solver_payload` can polish
   `gas_ln_n` with a NumPy least-squares step. The recorded KKT and support
   closure diagnostics describe the pre-polish state, and are not recomputed
   after an accepted polish. Result construction is therefore not a pure
   formatting operation.

3. **Production calls policies labelled diagnostic-only and default-off.**

   The support-selection and seed-policy reports hard-code
   `diagnostic_only=True`, `default_off=True`, and
   `production_behavior_change=False`, while the production lifecycle directly
   uses those decisions.

4. **Bucket preparation has two implementations.**

   Production builds fixed-support buckets in
   `api/condensate_equilibrium.py`; the experimental path uses a reusable
   builder in `optimize/fixed_support_v2_profile.py`. They can drift.

5. **`CondensateChemicalSetup` has multiple sources of truth.**

   It stores gas and condensate `ChemicalSetup` objects and repeats formula
   matrices, species, and elements. Current validation checks shapes and order
   but does not guarantee that repeated values equal the nested setup values.

6. **Gas equilibrium may be recomputed for one condensate layer.**

   Initial support discovery, fixed-support state construction, and a final
   gas-only outcome can each call the gas solver. The architecture should make
   reusable gas seed state explicit before attempting a performance change.

7. **Gas and condensate result contracts differ for more than naming.**

   Gas results are JAX pytrees and profile results are batched arrays.
   Condensate results contain host-side status, route, support names, and
   diagnostics; profile results contain a tuple of layer objects plus an
   optional batched-array mapping. Gas convergence is only visible when
   diagnostics are explicitly requested. These contracts must not be forced
   into one base result merely for visual symmetry.

8. **Preset compatibility is part of the solver input contract.**

   The public keyword `species_defalt_elements` contains a typo but is used by
   existing preset builders and must remain accepted if a corrected
   `species_default_elements` spelling is added. Species and element ordering
   are numerical contracts, including duplicate FastChem4 condensate slots.
   The older FastChem condensate builder can also construct its condensate
   setup from a different default gas setup than the combined builder's custom
   gas setup. Preset cleanup therefore needs explicit gas-setup injection and
   order-validation tests, not a mechanical rename.

9. **Gas autodiff has a narrow, route-specific contract.**

   The ordinary gas solve uses a custom reverse-mode VJP for temperature,
   normalized log pressure, and the element vector. Initial guesses are stopped,
   while the formula matrix, thermochemistry callable, tolerances, and iteration
   limit are non-differentiable arguments. Forward-mode `jax.jvp` / `jacfwd` is
   not supported by this `custom_vjp`. The diagnostics route calls the core
   iteration directly and does not have the same custom-VJP contract.

## 3. Architecture principles

### 3.1 Symmetry belongs at the capability boundary

Users should find the two capabilities through parallel namespaces and verbs:

```python
from exogibbs.api import condensate, gas

gas_result = gas.solve(...)
gas_profile = gas.solve_profile(...)

condensate_result = condensate.solve(...)
condensate_profile = condensate.solve_profile(...)
```

The same namespace pattern does not imply identical algorithms, profile
methods, transformability, setup types, or result containers.

Do not introduce a single `solve(..., condensates=True)` entry point. It would
hide materially different setup, execution, result, and failure contracts.

### 3.2 Internal dependency is deliberately asymmetric

Arrows below mean “depends on”:

```text
api.gas -----------------------> equilibrium.gas
api.condensate ----------------> equilibrium.condensate

equilibrium.condensate --------> equilibrium.gas
equilibrium.condensate --------> equilibrium.condensate.fixed_support

equilibrium.gas ---------------> thermo models
equilibrium.condensate --------> thermo models
presets -----------------------> thermo models
```

Forbidden directions are:

- `equilibrium.* -> api.*`;
- `presets.* -> api.*`;
- `equilibrium.gas -> equilibrium.condensate`;
- `fixed_support -> lifecycle`, public API, presets, or benchmark cases;
- production modules -> `tests`, `benchmarks`, or historical evidence.

The condensate layer must call the internal gas capability, not the public gas
facade. This keeps the API replaceable and removes API-to-API implementation
coupling.

### 3.3 Feature ownership precedes generic optimization ownership

Gas Newton code belongs to the gas-equilibrium feature even though it performs
optimization. Fixed-support PDIPM code belongs to condensate equilibrium even
though it is also an optimizer. A generic `optimize/` directory is useful only
for genuinely reusable optimization machinery; it must not be the default home
for feature-specific algorithms.

### 3.4 Share only identical concepts

The two capabilities may share thermochemical data models, pressure
normalization, and narrowly defined validation helpers. They must not share a
base `Options`, `Result`, `Initializer`, or `ProfileSolver` solely because both
sides have objects with those names.

Create a shared helper only when:

1. at least two production call sites need it;
2. the semantics and units are identical;
3. neither feature needs conditionals for the other feature;
4. its dependency direction remains clear.

There should be no initial `equilibrium/common/` dumping ground.

### 3.5 Separate host orchestration from compiled kernels

The following boundary is part of the design:

| Component | Execution model |
|---|---|
| Ordinary gas numerical kernel | JIT/vmap compatible; custom reverse-mode VJP for T, log-pressure, and b |
| Gas cold/hot profile scheduling | JAX `vmap` / `lax.scan` |
| Condensate support lifecycle | Host-side orchestration |
| Fixed-support numerical kernel and bucket solve | Compiled JAX |
| Condensate acceptance and reporting | Host-side unless explicitly proven otherwise |

Host-only conversion must occur at named boundary functions. A compiled kernel
must not gain hidden NumPy, SciPy, Python `float`, or `device_get` dependencies.

### 3.6 A move and a behavior change are separate changes

Compatibility aliases are allowed during migration. Numerical cleanup,
deduplication, cache redesign, convergence-contract changes, and correctness
fixes receive their own tests and commits after the relevant code has a clear
owner.

In particular:

- preserve `head_v2`, route version `v2.0`, and preset
  `validated_2026_07`;
- do not restore an executable v1 path or fallback;
- preserve module-scope custom-VJP callable identity during gas moves;
- preserve the rule that support discovery is outside a fixed-support solve;
- do not combine an internal package rename with a solver formula change.

## 4. Target public API

### 4.1 Canonical modules

Add two explicit, stable modules:

```text
exogibbs.api.gas
exogibbs.api.condensate
```

Each exports `solve` and `solve_profile`. During the refactor, existing
descriptive type names can be re-exported unchanged:

```text
api.gas
  solve
  solve_profile
  EquilibriumOptions
  EquilibriumInit
  EquilibriumInitRequest
  EquilibriumInitializer
  EquilibriumResult
  DefaultEquilibriumInitializer
  GridEquilibriumInitializer
  LearnedEquilibriumInitializer

api.condensate
  solve
  solve_profile
  CondensateChemicalSetup
  CondensateEquilibriumOptions
  CondensateEquilibriumInit
  CondensateEquilibriumInitRequest
  CondensateEquilibriumInitializer
  CondensateEquilibriumResult
  CondensateEquilibriumProfileResult
  DefaultCondensateEquilibriumInitializer
  CondensateFixedSupportV2Preset
  CondensateProfileMethod
  HEAD_ROUTE_V2
  CONDENSATE_HEAD_V2_ROUTE_NAME
  CONDENSATE_HEAD_V2_ROUTE_VERSION
  FIXED_SUPPORT_V2_VALIDATED_PRESET
  build_condensate_chemical_setup
  validate_condensate_chemical_setup
```

Short aliases such as `gas.Options` are optional convenience work and are not
required for architectural symmetry. This list describes the production-facing
canonical surface. Existing direct-module helpers and experimental symbols are
classified separately during the compatibility inventory.

### 4.2 Compatibility modules

The existing documented direct imports remain valid throughout this refactor:

| Existing path | Compatibility mapping |
|---|---|
| `exogibbs.api.equilibrium.equilibrium` | `api.gas.solve` |
| `exogibbs.api.equilibrium.equilibrium_profile` | `api.gas.solve_profile` |
| `exogibbs.api.condensate_equilibrium.condensate_equilibrium` | `api.condensate.solve` |
| `exogibbs.api.condensate_equilibrium.condensate_equilibrium_profile` | `api.condensate.solve_profile` |
| `exogibbs.api.chemistry.*` | Re-export from the new owning model modules |

This table shows the principal solver mappings, not the complete inventory.
Before umbrella normalization, classify every current name in
`exogibbs.api.__all__`, both direct API modules' `__all__`, active
documentation, and tests as one of:

- preserved non-colliding umbrella export;
- preserved direct-module compatibility export;
- canonical gas or condensate export;
- experimental export;
- developer-only symbol requiring an explicit removal decision.

In particular, the existing grid types/functions, setup/result/options types,
route constants, builders, validators, and non-colliding umbrella exports stay
available until that inventory explicitly says otherwise.

Compatibility modules should use direct aliases where possible so old and new
imports resolve to the same function or type object. Removal of these paths is
not part of this refactor and requires a separate release-compatibility
decision.

The umbrella package follows normal Python package semantics:

- `exogibbs.api.gas`, `exogibbs.api.condensate`,
  `exogibbs.api.equilibrium`, and
  `exogibbs.api.condensate_equilibrium` are modules;
- no umbrella attribute with one of those names is also a function;
- `__all__` contains only deterministic, import-order-independent values;
- importing `exogibbs.api` alone does not eagerly load the heavy condensate
  implementation;
- fresh-process tests cover all supported import orders.

The current ambiguous first-access behavior of
`api.condensate_equilibrium` is not a compatibility contract worth
preserving. Its normalization must nevertheless be released as an explicit API
change with tests and documentation.

### 4.3 Experimental API

Prepared fixed-support research plans are not part of the two production
facades. Their canonical user namespace should be explicit, for example:

```text
exogibbs.experimental.condensate
```

Existing imports from `api.condensate_equilibrium` can remain aliases during
migration. Production modules must not branch into experimental behavior.

### 4.4 Behavioral contracts during moves

The architecture move preserves these current semantics until a dedicated
behavior change is approved:

- `support_indices` supplied to the production condensate API is initial/base
  support and may be expanded; it is not a strict fixed-support request;
- the one-layer condensate API runs the same lifecycle as a length-one profile;
- condensate profiles accept `auto` / `vmap_cold` and reject gas-style hot
  scans;
- the current condensate profile does not pass a previous solution to its
  initializer, even though the initializer protocol contains that field;
- an accepted active-support state currently requires fixed-support
  convergence, support closure, independent KKT acceptance, and finite final
  state values before result polishing and the full-budget gate;
- lifecycle failure outcomes, acceptance tiers, statuses, warnings, electron
  handling, and disabled-gate behavior remain stable during mechanical moves.

## 5. Target source structure

The relevant portion of the target tree is feature-oriented. Unshown support
packages remain in place. The exact number of files may be adjusted to avoid
tiny modules, but the ownership boundaries must remain.

```text
src/exogibbs/
├── api/
│   ├── __init__.py
│   ├── gas.py
│   ├── condensate.py
│   ├── equilibrium.py                 # compatibility facade
│   ├── condensate_equilibrium.py      # compatibility facade
│   ├── chemistry.py                   # compatibility facade
│   ├── equilibrium_grid.py            # compatibility facade
│   └── potential.py                   # compatibility facade
│
├── thermo/
│   ├── models.py                      # ChemicalSetup and typed thermo data
│   ├── composition.py                 # element-vector helpers
│   ├── gibbs.py
│   ├── potential.py                   # phase-aware Gibbs energy utilities
│   └── stoichiometry.py
│
├── equilibrium/
│   ├── gas/
│   │   ├── types.py                   # state, options, init, result
│   │   ├── initialization.py
│   │   ├── solve.py                   # one-layer application service
│   │   ├── profile.py                 # vmap/scan scheduling
│   │   ├── grid/
│   │   │   ├── types.py
│   │   │   ├── interpolation.py
│   │   │   ├── storage.py
│   │   │   └── build.py
│   │   └── kernel/
│   │       ├── equations.py
│   │       ├── solver.py
│   │       ├── autodiff.py
│   │       └── diagnostics.py
│   │
│   └── condensate/
│       ├── types.py                   # options, init, public results
│       ├── setup.py                   # one validated source of truth
│       ├── solve.py                   # one-layer/profile application service
│       ├── initialization.py
│       ├── support.py                 # discovery, driving, seed decisions
│       ├── lifecycle.py               # host-side monotone support lifecycle
│       ├── acceptance.py              # explicit post-solve acceptance/polish
│       ├── results.py                 # pure result construction
│       ├── policy.py                  # production lifecycle policy
│       └── fixed_support/
│           ├── types.py
│           ├── problem.py
│           ├── filter.py
│           ├── linear_solver.py
│           ├── normal.py
│           ├── soc.py
│           ├── restoration.py
│           ├── return_map.py
│           ├── controller.py
│           ├── continuation.py
│           └── batch.py               # sole bucket builder/runner
│
├── experimental/
│   └── condensate.py
│
├── presets/
├── io/
└── utils/
```

`fixed_support_v2` can lose `_v2` only after all runtime and test imports have
migrated. Versioned public route metadata remains unchanged. Old
`exogibbs.optimize.*` paths may temporarily re-export the new implementations;
new production code must not import those compatibility paths.

### 5.1 Ownership contracts

| Owner | Owns | Must not own |
|---|---|---|
| `api.gas` | Stable gas names and documentation | Newton equations or profile caches |
| `api.condensate` | Stable condensate names and documentation | Support rounds or fixed-support math |
| `thermo.models` | Thermochemical input models and typed validity data | Solver policy |
| `gas.solve` | Input validation, kernel invocation, result assembly | Condensate decisions |
| `gas.profile` | Cold/hot layer scheduling and prepared/cache lifecycle | Newton formulas |
| `gas.kernel` | Gas numerical solve, residuals, custom VJP | API or preset imports |
| `condensate.support` | One sign convention for discovery and closure driving; seed decisions | Fixed-support iterations |
| `condensate.lifecycle` | Gas seed reuse, monotone support rounds, terminal outcomes | PDIPM iteration details |
| `condensate.fixed_support` | Exactly one fixed-support problem and batched execution | Support discovery or expansion |
| `condensate.acceptance` | Named transformations and post-transform validation | Object formatting disguised as numerical work |
| `condensate.results` | Pure conversion of an accepted state to result objects | Least-squares polishing |

The target `fixed_support.batch` returns active-support solver state, numerical
status, KKT components, and execution diagnostics. It does not receive the full
inactive-condensate catalog and does not decide closure or expansion.
`condensate.support` evaluates inactive-phase driving and closure from the
returned state; `condensate.lifecycle` decides whether to terminate or expand.
The current `run_prepared_profile_v2` combines both responsibilities, so this
boundary must be extracted before that adapter is moved wholesale.

### 5.2 Data-model rules

`ChemicalSetup.metadata` remains provenance and descriptive information. Any
value that changes numerical eligibility or solver behavior becomes a typed,
validated field. In particular, condensate temperature validity belongs in the
thermochemistry model defined in `thermo.models` and is validated for
condensate use by `condensate.setup`.

`CondensateChemicalSetup` should have one source of truth. Formula matrices and
species can be validated properties of its gas and condensate components rather
than independent copied fields. Properties alone do not preserve the current
public dataclass constructor, which accepts every duplicated field. During this
refactor, either retain that constructor with strict equality validation or
adapt it into a normalized internal setup. Removing constructor fields is a
separate breaking API change.

Introduce an internal gas seed/result carrier if the condensate lifecycle needs
gas log amounts, total amount, element potential, and stationarity source
together. This makes reuse explicit and prevents accidental repeated gas solves.
It is an internal contract, not another public result type.

### 5.3 Support and acceptance rules

Use one documented driving convention throughout support discovery and closure.
A recommended convention is:

```text
condensation_driving = A_cond.T @ element_potential - h_cond
```

Positive values then mean thermodynamically favorable condensation. If a
kernel naturally computes the negative quantity, convert at its boundary and
name it explicitly.

Operational selection returns an operational decision. Diagnostic reporting
wraps that decision; it must not hard-code “default off” for production calls.

Post-solve polishing is a named state transition, not result construction. An
accepted transformed state must either:

1. have stationarity, equality, total-density, and support-closure checks
   recomputed; or
2. be clearly reported as a distinct acceptance contract whose diagnostics do
   not claim to describe the transformed state.

Choosing between these is a scientific correctness decision and should be made
in a dedicated change.

## 6. Intended symmetry and intentional asymmetry

| Dimension | Gas | Condensate | Policy |
|---|---|---|---|
| Canonical module | `api.gas` | `api.condensate` | Symmetric |
| One-layer verb | `solve` | `solve` | Symmetric |
| Profile verb | `solve_profile` | `solve_profile` | Symmetric |
| Shared leading inputs | setup, T, P, b, Pref | setup, T, P, b, Pref | Align where meaningful |
| Setup model | Gas thermochemistry | Gas + condensate thermochemistry | Intentionally different |
| Numerical implementation | Gas Newton/CEA-style solve | Gas seed + support lifecycle + fixed-support PDIPM | Intentionally different |
| Profile implementation | JAX cold/hot scheduling | Host lifecycle over JAX buckets | Intentionally different |
| Result shape | JAX pytree, naturally batched | Layer objects plus optional batched arrays | Intentionally different |
| Autodiff contract | Ordinary route supports custom reverse-mode VJP for T/logP/b; no forward-mode JVP | Not currently a public whole-lifecycle contract | Intentionally different |
| Dependency | Independent foundation | Depends on gas | Intentionally asymmetric |

This table is the guardrail against two opposite mistakes: hiding the gas
feature behind generic names, and forcing fundamentally different runtimes
into an artificial common interface.

## 7. Migration plan

Each wave should be independently reviewable and leave all supported imports
working. Mechanical moves and behavior changes must not share a commit.

### Wave 0: characterization and dependency guardrails

1. Add fresh-process import-contract tests for direct imports, from-imports,
   star imports, both child-module import orders, and the existing lazy-load
   behavior of `import exogibbs.api`.
2. Inventory every umbrella and direct-module export, including grid
   types/functions, initializers, setup/result builders, route constants, and
   experimental names. Decide which direct `optimize.*` names, if any, are
   supported outside tests.
3. Add tests for gas non-convergence reporting, invalid setup/init shapes,
   profile ordering/cache behavior, and the current autodiff boundary:
   - public reverse-mode gradients with respect to T, P/logP, and b;
   - `vmap` of the ordinary differentiated route;
   - stopped gradients for initialization;
   - expected forward-mode JVP rejection;
   - separate characterization of the diagnostics route.
4. Add condensate characterization tests for:
   - `metadata=None` and typed temperature validity;
   - duplicated setup values;
   - pre/post-polish diagnostics;
   - production support-policy flags;
   - production versus experimental bucket preparation parity;
   - all acceptance predicates and lifecycle failure outcomes;
   - acceptance tier, status, warnings, electron handling, and disabled-gate
     behavior;
   - initial/base support expansion and rejected hot-scan methods.
5. Add a small deterministic end-to-end case through a real condensate preset,
   public production API, and real fixed-support kernel.
6. Add an offline dependency-boundary test using the Python AST or an existing
   repository tool; do not add a network-resolved dependency solely for this.

### Wave 1: canonical public facades

1. Add `api.gas` and `api.condensate` as thin facades over current behavior.
2. Add `solve` and `solve_profile` aliases without changing old direct modules.
3. Make `exogibbs.api` deterministic and module-valued for child-module names.
4. Update the README and active user documentation to show the new canonical
   imports and the old compatibility imports.
5. Keep this wave free of solver moves.

The umbrella normalization is the only intentional API behavior change in this
wave and must be called out in release notes.

### Wave 2: move models and setup below the API

Perform this wave as separate mechanical and model-migration commits:

1. **Wave 2a — neutral model identity move**
   - move `ChemicalSetup` to `thermo.models`;
   - move element-vector helpers to `thermo.composition`;
   - move `ThermoState` to `equilibrium.gas.types`;
   - re-export the same objects from `api.chemistry`;
   - migrate every consumer, including the condensate experimental plan.
2. **Wave 2b — condensate setup seam**
   - extract `CondensateChemicalSetup`,
     `build_condensate_chemical_setup`, and its validator into
     `equilibrium.condensate.setup` without changing their contract;
   - preserve the old public constructor and API re-exports.
3. **Wave 2c — dependency inversion removal**
   - update gas and condensate presets and implementations to import owning
     model/setup modules, never the API.
4. **Wave 2d — typed validity correctness change**
   - introduce typed condensate temperature-validity data and compatibility
     accessors in a dedicated, characterized change;
   - explicitly decide and test the new behavior for `metadata=None`;
   - normalize duplicated setup values only while preserving the legacy
     constructor or through a separately released breaking change.

This wave removes the dependency inversion required for both solver moves.

### Wave 3: extract the gas feature

Use four ordered subwaves so a parity failure has one likely owner:

1. **Wave 3a — equations and primal core**
   - isolate and move live residual, linear-system, step, and while-loop code;
   - keep `optimize/minimize.py` re-exports for every inventoried supported
     solver/diagnostic name;
   - do not move custom autodiff or profile orchestration yet.
2. **Wave 3b — reverse-mode adapter**
   - move the module-scope custom-VJP object and its forward/backward rules;
   - verify reverse-mode T/P/b and `vmap(grad)` parity;
   - retain the expected JVP rejection and diagnostics-route distinction.
3. **Wave 3c — one-layer application service**
   - move one-layer validation, default initialization, options, and result
     assembly into `equilibrium.gas`;
   - keep `GridEquilibriumInitializer` in the compatibility/API layer until the
     grid implementation moves in Wave 6, so internal code never imports
     `api.equilibrium_grid`.
4. **Wave 3d — profile scheduling**
   - move vmap/scan scheduling and cache ownership last;
   - account for cached closures, callable identity, and old monkeypatch seams
     explicitly rather than treating the old API module as a simple alias.

Keep `api/equilibrium.py` and `optimize/minimize.py` as compatibility facades
throughout these subwaves. Only after primal, reverse-mode, one-layer, and
profile parity are independently established should separate changes address:

- duplicate pressure normalization;
- duplicate step/linear-solve helpers;
- unused source-trace code;
- the unbounded object-ID profile cache;
- standard convergence fields in gas results.

### Wave 4: extract condensate orchestration

Perform extraction and deduplication in separate commits:

1. Extract the current lifecycle verbatim behind the old public API, preserving
   its state machine, bucket path, closure path, and result behavior.
2. Replace calls to `api.equilibrium` with the parity-tested internal gas
   capability.
3. Move types, initialization, and operational support decisions to their
   target owners without changing lifecycle decisions.
4. Prove production and experimental bucket-input/output parity, then replace
   the two builders with one implementation. This is not a mechanical move
   because their current initialization and validation paths differ.
5. Split inactive-condensate driving and closure out of
   `run_prepared_profile_v2`: the fixed-support batch runner returns its active
   state and KKT data; `condensate.support` evaluates full-catalog closure and
   `condensate.lifecycle` decides expansion.
6. Extract acceptance transformations and pure result construction in separate
   commits. Preserve pre/post-polish behavior until its correctness decision is
   reviewed.
7. Separate production policies from diagnostics and benchmark cases.
8. Move prepared experimental plans behind the experimental namespace while
   retaining compatibility aliases.
9. Address repeated gas solves only after a parity-tested internal gas-state
   carrier exists.

### Wave 5: relocate the fixed-support kernel

1. Move `optimize/fixed_support_v2/*` under
   `equilibrium/condensate/fixed_support/`.
2. Move `fixed_support_v2_profile.py` into that package as the batch adapter.
3. Retain old import aliases while internal call sites and tests migrate.
4. Drop `_v2` from internal filenames only after the old and new paths are
   proven equivalent.
5. Run the production GPU profile gate because module moves can change JAX
   tracing, compilation, and cache behavior even without formula changes.

### Wave 6: align supporting surfaces and remove migration scaffolding

1. Mirror `equilibrium/gas` and `equilibrium/condensate` under unit tests.
2. Split the gas grid module by model, interpolation, storage, and build
   responsibilities under the gas feature, then move
   `GridEquilibriumInitializer` below the API.
3. Align preset naming only through additive builders and compatibility
   aliases. Continue accepting `species_defalt_elements`, validate custom gas
   setup injection, and do not combine preset renames with solver moves.
4. Move benchmark-only curated profile definitions out of the installed runtime
   package unless a public-use audit justifies keeping them.
5. Organize examples and active docs into gas and condensate sections. Add one
   comparison page covering the parallel quick starts, shared inputs, different
   result shapes, profile methods, and JAX contracts.
6. Delete zero-reachability compatibility scaffolding only after a separate
   public-compatibility review.
7. Keep historical validation artifacts and recorded hashes unchanged.

## 8. Validation policy

### 8.1 Required checks by change type

| Change | Minimum validation |
|---|---|
| Documentation only | `git diff --check` and link/path review |
| Public facade | Fresh-process import/lazy-load matrix, export manifest, and API unit tests |
| Shared model move | Preset, API, serialization/type-identity, and full unit tests |
| Gas implementation move | Analytical gas tests, API tests, JIT/vmap/reverse-mode parity, expected-JVP test, full unit tests |
| Condensate lifecycle move | Lifecycle, policy, result-gate, real end-to-end, and full unit tests |
| Fixed-support import or implementation move | Fixed-support unit tests, production route contracts, full suite, A100 production profile gate |
| Numerical behavior change | Dedicated regression, independent invariant checks, benchmarks, and relevant production gate |

Use the current worktree explicitly:

```bash
env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q tests/unittests
```

For gas work:

```bash
env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q \
  tests/unittests/api/equilibrium_api_test.py \
  tests/unittests/api/equilibrium_profile_test.py \
  tests/unittests/api/equilibrium_jit_grad_test.py \
  tests/unittests/optimize/lagrange
```

For condensate orchestration and the fixed-support boundary:

```bash
env PYTHONPATH=/home/kawahara/exogibbs-condensate-cleanup/src \
  .venv/bin/python -m pytest -q \
  tests/unittests/api/condensate_equilibrium_test.py \
  tests/unittests/api/condensate_equilibrium_profile_test.py \
  tests/unittests/api/condensate_production_route_contract_test.py \
  tests/unittests/condensates \
  tests/unittests/optimize
```

The full suite is required before each completed implementation wave. The A100
gate described in `benchmarks/fixed_support_v2/README.md` is additionally
required for Waves 4 and 5 when runtime imports, lifecycle execution, compiled
callable identity, or fixed-support code changes.

### 8.2 Compatibility invariants

Tests should assert all of the following during migration:

- old and new public imports produce equivalent results;
- compatibility type re-exports preserve object identity where promised;
- import results do not depend on import order;
- `import exogibbs.api` preserves lazy loading of the condensate implementation;
- all inventoried non-colliding umbrella exports remain available;
- the ordinary gas route preserves JIT, vmap, scan, and reverse-mode gradients
  for T/P/b, while unsupported forward-mode and the diagnostics route retain
  their characterized behavior;
- condensate route name, version, preset, status, and diagnostics schema remain
  unchanged unless a separately reviewed schema migration says otherwise;
- species and element ordering, including intentional duplicate species slots,
  remain unchanged;
- existing preset keywords continue to work through compatibility aliases;
- fixed-support failure never selects a retired fallback;
- frozen historical artifact bytes and hashes are untouched.

## 9. Explicit non-goals

This architecture refactor does not:

- create a unified gas/condensate solver class;
- make the whole condensate lifecycle differentiable or JIT-compatible;
- broaden the gas custom-VJP contract to forward-mode autodiff;
- change the gas algorithm or fixed-support PDIPM mathematics;
- change production tolerances, support limits, route metadata, or preset;
- restore v1 runtime code;
- standardize gas and condensate result contents in the same move;
- remove documented compatibility modules;
- reorganize every historical document before solver ownership is clear.

## 10. Follow-up decisions that must remain separate

The audit exposed product and correctness questions, but they should not block
the package layout:

1. Whether gas results should always include convergence, iteration, and final
   residual fields.
2. Whether accepted gas polishing in condensate results must be followed by a
   complete KKT and support-closure recheck.
3. Whether gas profile caching should become an explicit prepared solver object
   instead of a module-global dictionary.
4. Whether direct imports from `exogibbs.optimize.*` are supported public API or
   developer-only implementation access.
5. Whether curated production cases belong to runtime presets, examples, or
   benchmark fixtures.
6. The release in which compatibility facades may eventually be deprecated.

Each decision needs its own contract tests and release note if it changes user
behavior.

## 11. Recommended first implementation slice

The first implementation PR should be deliberately small:

1. add the fresh-process import characterization tests;
2. record the complete umbrella/direct-module export classification;
3. add a dependency-boundary test;
4. add `api.gas` and `api.condensate` as additive thin modules;
5. document `gas.solve` / `condensate.solve`;
6. retain every existing direct module entry point and lazy-load behavior;
7. do not move solver code or change the umbrella collision in the same commit.

A following API-only PR can normalize the umbrella module behavior with an
explicit compatibility note. Only then should model and implementation moves
begin.

## 12. Completion criteria

The architecture refactor is complete when:

- users discover both capabilities through `api.gas` and `api.condensate`;
- `api/` contains facades and documentation, not numerical or lifecycle logic;
- presets and numerical code do not import `api/`;
- gas code is explicitly owned by `equilibrium.gas`;
- condensate lifecycle is explicitly owned by `equilibrium.condensate` and
  depends on the internal gas capability;
- fixed-support code owns exactly one fixed-support solve and no support
  discovery;
- inactive-phase closure is evaluated outside the fixed-support batch runner;
- production and experimental bucket preparation have one implementation;
- runtime numerical inputs are typed fields rather than optional metadata;
- result construction is pure and any numerical post-processing is explicit;
- source and test trees mirror the same ownership;
- supported old imports remain deterministic compatibility aliases;
- the full unit suite and required production gates pass.

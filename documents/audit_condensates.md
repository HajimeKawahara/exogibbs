• 1. Executive Summary

  The main condensate gap is not just algorithmic; it is structural. The gas-only path has a stable public API, explicit
  initializer/hot-start design, lightweight diagnostics, and deliberate transform boundaries in src/exogibbs/optimize/
  minimize.py:458 and src/exogibbs/api/equilibrium.py:316. The condensate path in use is a lower-level epsilon-scheduled
  core in src/exogibbs/optimize/pipm_rgie_cond.py:293, while src/exogibbs/optimize/minimize_cond.py:1 points at a
  different legacy implementation entirely.

  What to do first: fix visibility and API shape before changing numerics. Hot start is blocked mostly by missing
  solver/state abstractions, not by math. Compile stability risk is driven by nested local wrappers, direct driver-level
  jit(vmap(...)), debug branches inside traced code, and the split/ambiguous condensate entry points. Convergence is
  hard to diagnose because the production condensate solver only returns final state plus iteration count.

  2. Structural Diff: minimize_cond.py vs minimize.py

  - Gas-only has a clear layered structure:
    prepare, linear solve, finish solve, apply damped step, evaluate residual, then while_loop in src/exogibbs/optimize/
    minimize.py:185, src/exogibbs/optimize/minimize.py:267, src/exogibbs/optimize/minimize.py:458.
  - Condensate public file is only a shim to old PDIPM code in src/exogibbs/optimize/minimize_cond.py:1, while examples
    bypass it and call RGIE/PIPM directly in documents/ipynb/pipm/rgie/fastchem_cond_one.py:95 and documents/ipynb/pipm/
    rgie/fastchem_cond_prof.py:117.
  - Gas-only API is stable and user-facing: minimize_gibbs, minimize_gibbs_with_diagnostics, equilibrium,
    equilibrium_profile in src/exogibbs/optimize/minimize.py:627 and src/exogibbs/api/equilibrium.py:316.
  - Condensate API is solver-internal: requires epsilon, residual_crit, both gas and condensate formula matrices, and
    raw ln_nk/ln_mk/ln_ntot in src/exogibbs/optimize/pipm_rgie_cond.py:293.
  - Gas-only state is compact: ln_nk, ln_ntot, gk, An, residual, counter.
  - Condensate state adds ln_mk, Am, barrier-derived ln_sk = 2 ln_mk - epsilon, plus a more coupled reduced linear
    system in src/exogibbs/optimize/pipm_rgie_cond.py:18 and src/exogibbs/optimize/pipm_rgie_cond.py:168.
  - Gas-only initialization is abstracted through EquilibriumInit, EquilibriumInitializer, and profile carryover in src/
    exogibbs/api/equilibrium.py:53, src/exogibbs/api/equilibrium.py:238.
  - Condensate initialization is manual in drivers: either zeros or gas-only equilibrium seed, with ln_mk = 0 and no
    public initializer object in documents/ipynb/pipm/rgie/fastchem_cond_one.py:111 and documents/ipynb/pipm/rgie/
    fastchem_cond_prof.py:208.
  - Gas-only damping is encapsulated and simple via _cea_lambda in src/exogibbs/optimize/minimize.py:164.
  - Condensate damping is heuristic and multi-part: gas CEA step cap, condensate delta cap, and s_k cap in src/exogibbs/
    optimize/pipm_rgie_cond.py:241 with helpers in src/exogibbs/optimize/stepsize.py:7.
  - Gas-only returns diagnostics and also has a profiling helper in src/exogibbs/optimize/minimize.py:304 and src/
    exogibbs/optimize/minimize.py:664.
  - Condensate returns only (ln_nk, ln_mk, ln_ntot, counter) in the active solver in src/exogibbs/optimize/
    pipm_rgie_cond.py:397. No final residual is returned.
  - Gas-only has deliberate transform engineering: module-scope while_loop callables and module-scope custom_vjp wrapper
    in src/exogibbs/optimize/minimize.py:69 and src/exogibbs/optimize/minimize.py:541.
  - Condensate has only a raw while_loop; no public jit/custom_vjp boundary, and cond_fun/body_fun are recreated per
    call in src/exogibbs/optimize/pipm_rgie_cond.py:352.
  - Local/mechanical quality issue: final while_loop unpacking swaps An/Am names in src/exogibbs/optimize/
    pipm_rgie_cond.py:397. It is harmless here because they are not returned, but it signals weak structural hygiene.

  3. Call-Path Audit from fastchem_cond_one.py and fastchem_cond_prof.py

  - Single-layer path contracts gas/condensate formula matrices first, seeds from gas-only equilibrium or zeros, builds
    ThermoState, then runs an outer epsilon schedule with lax.fori_loop; each schedule step calls
    minimize_gibbs_cond_core with the previous (ln_nk, ln_mk, ln_ntot) as input in documents/ipynb/pipm/rgie/
    fastchem_cond_one.py:73, documents/ipynb/pipm/rgie/fastchem_cond_one.py:156.
  - Profile path defines a local per-layer wrapper minimize_gibbs_cond, does the same epsilon schedule inside that
    function, then applies vmap and jit over layers in documents/ipynb/pipm/rgie/fastchem_cond_prof.py:148 and
    documents/ipynb/pipm/rgie/fastchem_cond_prof.py:233.
  - Current profile behavior is cold-start across layers even when using gas-only initialization: each layer gets an
    independent gas-only seed computed in Python; no solved condensate state is propagated to the next layer in
    documents/ipynb/pipm/rgie/fastchem_cond_prof.py:208.
  - A future hot-start design can be inserted without major top-level breakage if the public layer-level API accepts an
    explicit condensate init object and the profile driver switches from vmap cold-start to scan carryover, mirroring
    gas-only src/exogibbs/api/equilibrium.py:396.

  4. Hot-Start Readiness Assessment

  - Already compatible: the active condensate core already accepts previous ln_nk, ln_mk, and ln_ntot, so the numerical
    kernel itself is warm-startable.
  - Main blocker: there is no public condensate result/init type analogous to EquilibriumInit/EquilibriumResult.
  - Main blocker: epsilon scheduling is externalized into the drivers, so “state” currently means both primal variables
    and schedule position. A reusable hot start needs a defined policy for whether to restart from epsilon_start or
    reuse a later-stage state.
  - Main blocker: profile path is vmap cold-start, not scan carryover.
  - Risk: carrying ln_mk directly across layers may be numerically fragile when phase activity changes sharply; likely
    manageable, but not inferable without running.

  5. Compile-Stability Assessment

  - Stronger gas-only design: module-scope transformed functions and cached scan-body identity reduce retracing pressure
    in src/exogibbs/optimize/minimize.py:541 and src/exogibbs/api/equilibrium.py:246.
  - Condensate risks:
    the practical entry point is a local nested function in the profile example, then vmap and jit are built on top of
    it in documents/ipynb/pipm/rgie/fastchem_cond_prof.py:148 and documents/ipynb/pipm/rgie/fastchem_cond_prof.py:233.
  - Condensate risks:
    debug_nan branches and jdebug.print live inside the traced solver body in src/exogibbs/optimize/
    pipm_rgie_cond.py:125 and src/exogibbs/optimize/pipm_rgie_cond.py:187.
  - Condensate risks:
    two different condensate implementations exist (pdipm_cond, pipm_gie_cond, pipm_rgie_cond) and the compatibility
    import points at the old one, which increases accidental transform drift and user confusion.
  - Condensate likely stable in shape terms within one chosen setup: array ranks look static, epsilon schedule length is
    static in examples, and while_loop carry shapes are fixed.
  - Uncertain without execution: whether repeated calls with identical shapes still retrace due to callable identity in
    real workloads.

  6. Convergence/Diagnostics Visibility Assessment

  - Gas-only already exposes n_iter, converged, hit_max_iter, final_residual, and has a profiling helper in src/
    exogibbs/optimize/minimize.py:304 and src/exogibbs/optimize/minimize.py:664.
  - Condensate exposes only counter; final residual, final lambda, barrier stage, and failure mode are hidden.
  - Current debug visibility is all-or-nothing jdebug.print, useful for NaN hunts but not for routine diagnosis.
  - The epsilon outer schedule is invisible in returned values, so you cannot tell whether difficulty comes from inner
    iteration stagnation or from schedule-stage progression.

  7. Ranked Staged Plan

  1. Phase 0: audit/instrumentation only.
     Target: add a condensate diagnostics-returning wrapper and a small profiling helper analogous to gas-only, without
     changing update math.
     Benefit: immediate visibility.
     Risk: low.
     Success: can inspect per-call n_iter, final residual, converged, hit_max_iter, epsilon used.
  2. Phase 1: unify public entry points.
     Target: stop the split-brain between src/exogibbs/optimize/minimize_cond.py:1 and the actually used solver; choose
     one public condensate core and make examples call it.
     Benefit: removes usability ambiguity and reduces maintenance mistakes.
     Risk: low to medium if old callers exist.
     Success: one documented condensate import path.
  3. Phase 2: introduce condensate init/result state objects.
     Target: add a small dataclass pair for ln_nk, ln_mk, ln_ntot, plus diagnostics.
     Benefit: makes hot start and API parity possible with minimal math change.
     Risk: low.
     Success: one-layer solve can accept explicit previous condensate solution cleanly.
  4. Phase 3: add profile hot-start scan path.
     Target: replace driver-level vmap cold-start with a scan_hot_from_bottom/top condensate profile wrapper.
     Benefit: addresses the largest usability/performance gap.
     Risk: medium because layer-to-layer phase changes may expose robustness issues.
     Success: profile API supports both cold and hot modes; iteration counts drop materially on smooth profiles.
  5. Phase 4: compile-stability cleanup.
     Target: move practical transformed wrappers to module scope, cache scan bodies if a public profile API is added,
     isolate debug printing from normal traced path.
     Benefit: better repeated-call stability.
     Risk: medium.
     Success: repeated same-shape calls avoid unexpected recompiles in benchmarks/logs.
  6. Phase 5: optional numerical cleanup.
     Target: revisit heuristic clipping and barrier-stage policy only after diagnostics show where failures occur.
     Benefit: actual robustness improvement, but only with evidence.
     Risk: highest because it can change behavior.
     Success: fewer stalled/NaN cases on known troublesome layers without regressing current matches.

  8. Recommended First Coding Task

  Add a condensate-side minimize_gibbs_cond_with_diagnostics wrapper around the active solver in src/exogibbs/optimize/
  pipm_rgie_cond.py:293, returning at least n_iter, final_residual, residual_crit, converged, hit_max_iter, and epsilon.

  That is the smallest safe next step because it preserves current math, clarifies whether the real pain is inner-loop
  convergence or outer epsilon scheduling, and creates the measurement surface needed before any hot-start or compile-
  stability refactor.

Curated default adoption, 2026-11-02
===================================

The public ``curated.chemsetup`` now selects the existing ``curated-c2h-v1``
candidate. ``curated_v1.chemsetup`` preserves the original preset source bytes,
data identity and thermochemistry. No coefficient dataset is overwritten.

The scientific reason is the superseded 1967 JANAF energy estimate and a
thermal-function discrepancy that persists after energy alignment. NASA CCH
thermal functions plus balanced ATcT 1.172 atomization energy address both
identified issues. Incomplete primary-source inspection and unquantified
high-temperature thermal uncertainty qualify this preference; they do not
establish that the old estimate is preferable. See
``documents/presets/curated_c2h.rst`` for the source reasoning and limitations.

The new execution cold-solves 75 conditions per network, all passing at
5e-14 tolerance: five temperatures, five pressures and three C/O budgets.
The two NPZ files retain every species' log amount, log mole fraction, h,
formula matrix, budget, conditions and convergence flags. The ion-inclusive
results reproduce the original candidate's strict log mole fractions exactly
in the recorded environment. Neutral-mode results pass numerical closure;
no additional independent-solver or physical validation is claimed.

Replay into a new directory::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/curated_default_adoption.py \
     --output-dir results/curated-default-replay

Historical source locations
----------------------------

``source/`` preserves seven original files from commit
``a541a7441a39b100c3ea5b5621f7723811c03b9c`` before explicit legacy imports
and receipt-path updates. These are source snapshots, not standalone replay
entry points. Run the current review scripts for replay; they explicitly use
``curated_v1`` when reproducing historical comparisons.

``mceg/reviews/curated_history.py`` resolves original receipt paths to these
snapshots or, for the original ``curated.py``, to the byte-identical
``curated_v1.py``. Historical receipt hashes and outputs are never rewritten.
Their checks establish the original execution identity, not current default
verification. The new receipt here separately pins the current implementation.

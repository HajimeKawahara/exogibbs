C2N2 bounded energy and thermal candidate evaluation
====================================================

Retaining curated is scientifically acceptable for this grid. The ATcT physical energy shift is only +294 J/mol in G/H, comparable to the quoted 0.31 kJ/mol molecular marginal uncertainty (not a reaction covariance). The large NASA thermal/fit difference is not evidence that NASA is more accurate. The energy-only alternative remains an optional small modern-energy update, not a mandatory fix.

The two alternatives are independent replacements on ``curated-ch2-v1``:
retained literal thermals plus ATcT energy, and NASA9 thermals plus the same
balanced ATcT energy. Production coefficients remain unchanged in this review.
Only ``C2N2`` changes in each candidate. Original source-review packets and
all prior execution receipts remain immutable.

Physical JANAF D0 is 2056794.000000 J/mol and ATcT 1.228 D0 is
2056500.000000 J/mol. Thus the retained candidate changes G/H by
294.000000 J/mol and leaves S/Cp exactly unchanged. The
formal R*a fit intercept (2064343.504231 J/mol)
is a separate quantity. Its source-fit residual is retained by the energy-only
candidate. No second ZPE, isotope, spin or zero-K fit extrapolation is applied.
The NASA control removes its own physical energy with native R=8.314510,
inserts ATcT D0 with SI R=8.31446261815324, and retains dimensionless NASA
molecular-minus-stoichiometric-atomic thermals. It also replaces the old fit
residual; it is not an isolated molecular thermal correction.

Actual computation
------------------

Each alternative receives 75 ion-inclusive (537 species) and 75 neutral
(421 species) fresh cold endpoints: 300 fresh endpoints in total. The same
150 archived current endpoints are reused only after all hashes, named-factory
potentials, species, matrices, budgets and conditions match. The grid spans
1500, 2000, 2500, 3000 and 4000 K; 1e-6, 1e-4, 0.01, 1 and 100 bar;
and C/O=0.5888436553555908, 1 and 1.5. This is a finite hot H-rich ideal-gas
grid, not an arbitrary planetary atmosphere validation.

There are 18 fresh tighter controls (current plus both alternatives, three
conditions per network), 10 interpolation-path controls, and six independent
ion-inclusive FastChem executions (current plus both alternatives, two selected
conditions). Baseline repeats are not additional distinct conditions.
All ExoGibbs solves converge; unfloored element/charge/mass-action and target-only
potential checks pass. Maximum FastChem difference is 2.33e-08 dex.

.. list-table:: Solved target mole-fraction ratios and resolved same-model Gibbs checks
   :header-rows: 1

   * - Network / candidate
     - x(candidate) / x(current)
     - Resolved Gibbs cases
   * - ions / retained
     - 0.9767023–0.991198959
     - 20/75
   * - ions / nasa
     - 1.06780594–1.23509202
     - 30/75
   * - neutral / retained
     - 0.9767023–0.991198959
     - 20/75
   * - neutral / nasa
     - 1.06780594–1.23509202
     - 30/75

Unresolved same-model Gibbs differences remain below the displayed numerical
scale and are not relabeled as proven ordering. The path, stricter reference,
formation-equation and independent-solver checks support finite numerical
consistency, not physical source ranking or universal Gibbs resolution.
For an extremely small trace species, displayed changes in other species may
be numerical noise larger than its elemental redistribution; no bulk-atmosphere
effect is inferred from those values.

A high-temperature state or evaluated thermal model that resolves the 10–20% JANAF/NASA thermal difference is still missing. Cyanogen NCCN is distinct from CNN, NCN and CNC. Complete reaction covariance and physical abundance accuracy remain
unknown. These limits do not require keeping the scoped investigation PR open.
A future production update, if selected, needs its own packaged factory and
production verification. The candidate result alone is not a production update.

Reproduction and records
------------------------

The continuous source check uses 2752 integer and off-grid temperatures and
independent G/H/S/Cp expressions, then scalar/JAX derivatives of one potential.
The dimensional G/H agreement bound is 2e-8 J/mol (O3 NASA subtraction reaches
1.27e-8 J/mol through floating-point cancellation); this numerical bound is
not a physical uncertainty. Source pins, actual commands and output hashes are
in the candidate and impact receipts. Tests replay archives without FastChem
or network access.

.. code-block:: console

   PYTHONPATH=src python mceg/reviews/pending_thermal_gap_inputs.py c2n2
   PYTHONPATH=src python mceg/reviews/pending_equilibrium_impact.py --definition tests/data/validation/c2n2_candidate/20261106/candidate.json --output-dir results/c2n2_pending_impact --fastchem-executable /path/to/fastchem
   PYTHONPATH=src pytest -q tests/model_evidence/c2n2_candidate_test.py

The candidate constructor refuses to overwrite an executed receipt. The original
source review :doc:`c2n2_source_review` contains the inspected primary-source
URLs, source identity and missing thermal evidence. MCEG separately records
construction, finite response, the scoped recommendation and unresolved claims.
Related: `#398 <https://github.com/HajimeKawahara/exogibbs/pull/398>`_,
`#308 <https://github.com/HajimeKawahara/exogibbs/issues/308>`_.

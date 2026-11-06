C3O2 bounded energy and thermal candidate evaluation
====================================================

Prefer the same-version ATcT physical energy correction as a scoped future update, preserving the existing thermal model and fit residual. The source cycle supports G/H +1698 J/mol; the NASA alternative changes the thermal/gauge construction and can reverse the response, without demonstrated physical superiority. Negligible abundance in this grid does not by itself validate or invalidate its energy source.

The two alternatives are independent replacements on ``curated-ch2-v1``:
retained literal thermals plus ATcT energy, and NASA9 thermals plus the same
balanced ATcT energy. Production coefficients remain unchanged in this review.
Only ``C3O2`` changes in each candidate. Original source-review packets and
all prior execution receipts remain immutable.

Physical JANAF D0 is 2723954.000000 J/mol and ATcT 1.228 D0 is
2722256.000000 J/mol. Thus the retained candidate changes G/H by
1698.000000 J/mol and leaves S/Cp exactly unchanged. The
formal R*a fit intercept (2735576.082221 J/mol)
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
potential checks pass. Maximum FastChem difference is 1.18e-08 dex.

.. list-table:: Solved target mole-fraction ratios and resolved same-model Gibbs checks
   :header-rows: 1

   * - Network / candidate
     - x(candidate) / x(current)
     - Resolved Gibbs cases
   * - ions / retained
     - 0.872713194–0.950225823
     - 1/75
   * - ions / nasa
     - 0.948185758–1.07024923
     - 2/75
   * - neutral / retained
     - 0.872713194–0.950225823
     - 1/75
   * - neutral / nasa
     - 0.948185758–1.07024923
     - 2/75

Unresolved same-model Gibbs differences remain below the displayed numerical
scale and are not relabeled as proven ordering. The path, stricter reference,
formation-equation and independent-solver checks support finite numerical
consistency, not physical source ranking or universal Gibbs resolution.
For an extremely small trace species, displayed changes in other species may
be numerical noise larger than its elemental redistribution; no bulk-atmosphere
effect is inferred from those values.

The soft bending/large-amplitude mode remains unresolved by a new complete hot partition function. Do not fabricate an RRHO correction from incomplete mode data or extrapolate low-temperature partition evidence. Complete reaction covariance and physical abundance accuracy remain
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

   PYTHONPATH=src python mceg/reviews/pending_thermal_gap_inputs.py c3o2
   PYTHONPATH=src python mceg/reviews/pending_equilibrium_impact.py --definition tests/data/validation/c3o2_candidate/20261106/candidate.json --output-dir results/c3o2_pending_impact --fastchem-executable /path/to/fastchem
   PYTHONPATH=src pytest -q tests/model_evidence/c3o2_candidate_test.py

The candidate constructor refuses to overwrite an executed receipt. The original
source review :doc:`c3o2_source_review` contains the inspected primary-source
URLs, source identity and missing thermal evidence. MCEG separately records
construction, finite response, the scoped recommendation and unresolved claims.
Related: `#400 <https://github.com/HajimeKawahara/exogibbs/pull/400>`_,
`#308 <https://github.com/HajimeKawahara/exogibbs/issues/308>`_.

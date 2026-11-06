NCN bounded candidates and fixed-network substitution
=====================================================

Recommend the current evaluated ATcT energy with retained thermals. Keep NASA as a separate sensitivity and the 2026 measurement as corroboration. Full excited-state thermal accuracy remains unknown. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral linear N-C-N, X3Sigma-g-minus, C1N2_ncn, symmetry 2; not CNN.

The ATcT 1.228 D0=1201.350 kJ/mol is +20.792 relative to JANAF. The inspected 2026 threshold study gives Hf0=450.5+/-3.5 kJ/mol, compatible with the evaluated 451.33. This is corroborating evidence with its stated inspection scope, not a refit of the complete experiment.

Original physical D0=1180.558000000 and selected D0=1201.350000000 kJ/mol. The retained G/H shift is -20.792000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

Energy-aligned JANAF/NASA differences are approximately5-6%. JANAF also estimates a high singlet state. The NASA alternative is independently normalized; neither CNN constants nor an outdated2013 ATcT energy is used.

The original :doc:`ncn_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

The NASA alternative uses the exact molecule-minus-neutral-atoms NASA9 reaction. It removes source D0 with native R=8.314510 and inserts the selected physical energy with SI R=8.31446261815324, once. Its molecular/atomic thermal gauge and removed fit residual make it a full reference-model sensitivity, not a pure molecular correction or uncertainty interval.

Calculation scope and counts
----------------------------

All variants independently start from named ``curated-ch2-v1`` after HCP/C2H2/CH2 adoption. Each uses 75 conditions in both 537-species ion and 421-species neutral networks. The 150 common current endpoints are reused once with file hashes, named factory potentials, species, formula matrix and elemental budgets verified.

Temperatures are 1500/2000/2500/3000/4000 K,  pressures 1e-6/1e-4/0.01/1/100 bar,  and C/O=0.5888436553555908/1/1.5. Only the explicitly declared target species change; the whole fixed gas network re-equilibrates.

There are **300 fresh main endpoints**, 18 tighter cold solves, 10 path solves, and 0 selected-alternative solves. FastChem executes 6 times at two ion-network conditions, including current and every main variant. These are actual executions, not that many distinct conditions. No neutral FastChem run is implied.

Finite composition response
---------------------------

.. list-table:: Solved target response; shares are maxima on each model grid
   :header-rows: 1

   * - Network/model
     - Minimum/maximum x over current
     - Maximum N share (%)
   * - ions/current
     - 1 / 1
     - 2.28064998e-05
   * - ions/retained
     - 1.8685737 / 5.2969713
     - 4.26156257e-05
   * - ions/nasa
     - 1.75793662 / 4.97968233
     - 4.00923811e-05
   * - neutral/current
     - 1 / 1
     - 2.28098996e-05
   * - neutral/retained
     - 1.8685737 / 5.2969713
     - 4.26219785e-05
   * - neutral/nasa
     - 1.75793662 / 4.97968233
     - 4.00983578e-05

At the ions current maximum-N-share condition (4000 K, 100 bar, C/O=1.5), the same-condition percentages are 2.28064998e-05 / 4.26156257e-05 / 4.00923811e-05 in order current, retained, nasa. The table above allows each model its own maximum.

At the neutral current maximum-N-share condition (4000 K, 100 bar, C/O=1.5), the same-condition percentages are 2.28098996e-05 / 4.26219785e-05 / 4.00983578e-05 in order current, retained, nasa. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C1N2_ncn 9.11474e-07 percentage points at 4000 K/100 bar/C/O=1.5, C1H1N1_hcn 5.98033e-07 percentage points at 4000 K/100 bar/C/O=1.5, C1 2.69157e-07 percentage points at 4000 K/1 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/retained, the largest absolute N-allocation changes are C1N2_ncn 1.98091e-05 percentage points at 4000 K/100 bar/C/O=1.5, N2 1.06316e-05 percentage points at 4000 K/100 bar/C/O=1.5, C1H1N1_hcn 6.49854e-06 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute C-allocation changes are C1N2_ncn 7.95372e-07 percentage points at 4000 K/100 bar/C/O=1.5, C1H1N1_hcn 5.21856e-07 percentage points at 4000 K/100 bar/C/O=1.5, C1 2.34873e-07 percentage points at 4000 K/1 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute N-allocation changes are C1N2_ncn 1.72859e-05 percentage points at 4000 K/100 bar/C/O=1.5, N2 9.27737e-06 percentage points at 4000 K/100 bar/C/O=1.5, C1H1N1_hcn 5.67077e-06 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 1.46e-09 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.3e-13, and formation residual 1.19e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.18e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 31/75, ions nasa 31/75, neutral retained 31/75, neutral nasa 31/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend the current evaluated ATcT energy with retained thermals. Keep NASA as a separate sensitivity and the 2026 measurement as corroboration. Full excited-state thermal accuracy remains unknown.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1016/j.proci.2026.106251 <https://doi.org/10.1016/j.proci.2026.106251>`_.
* `DOI 10.1016/j.proci.2012.06.128 <https://doi.org/10.1016/j.proci.2012.06.128>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/ncn_candidate_test.py

The definition is ``tests/data/validation/ncn_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

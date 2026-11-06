C4N2 bounded candidates and fixed-network substitution
======================================================

A bounded corrected-energy update using the explicitly selected NASA physical anchor is supported by the documented combustion correction. Prefer retaining thermals for that step; the second full NASA model is a sensitivity. This trace-species update has low immediate priority on the tested grid. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral linear singlet N-C-C-C-C-N (dicyanoacetylene), rotational symmetry 2.

NIST reports the 1963 correction of the original combustion result: Hf298 529.3 kJ/mol instead of the older533.46. NASA/TRC uses 529.2. Here the explicitly available NASA balanced physical D0=3262.140965 kJ/mol is the corrected-energy control, +4.358965 relative to JANAF; it is not called an ATcT determination or exact529.3 result.

Original physical D0=3257.782000000 and selected D0=3262.140965000 kJ/mol. The retained G/H shift is -4.358965000 kJ/mol; S/Cp and the complete fit residual are unchanged.

The NASA/JANAF thermal contrast remains separate from the documented combustion correction. The full correction text was not reanalysed, and no new high-temperature partition function is inferred from the structure paper.

The original :doc:`c4n2_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

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
     - 5.8042945e-09
   * - ions/retained
     - 1.14004273 / 1.41837047
     - 6.91264542e-09
   * - ions/nasa
     - 1.16218473 / 1.44225552
     - 7.25147653e-09
   * - neutral/current
     - 1 / 1
     - 5.80461789e-09
   * - neutral/retained
     - 1.14004273 / 1.41837047
     - 6.91303056e-09
   * - neutral/nasa
     - 1.16218473 / 1.44225552
     - 7.25188054e-09

At the ions current maximum-N-share condition (3000 K, 1 bar, C/O=1.5), the same-condition percentages are 5.8042945e-09 / 6.91264542e-09 / 7.25147653e-09 in order current, retained, nasa. The table above allows each model its own maximum.

At the neutral current maximum-N-share condition (3000 K, 1 bar, C/O=1.5), the same-condition percentages are 5.80461789e-09 / 6.91303056e-09 / 7.25188054e-09 in order current, retained, nasa. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C4N2 2.03993e-10 percentage points at 3000 K/1 bar/C/O=1.5, C2H2 1.37873e-10 percentage points at 3000 K/1 bar/C/O=1.5, C3H1 4.70013e-11 percentage points at 2000 K/1e-06 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/retained, the largest absolute N-allocation changes are C4N2 1.10835e-09 percentage points at 3000 K/1 bar/C/O=1.5, N2 5.36438e-10 percentage points at 3000 K/1 bar/C/O=1.5, C1H1N1_hcn 4.85822e-10 percentage points at 3000 K/1 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute C-allocation changes are C4N2 2.66356e-10 percentage points at 3000 K/1 bar/C/O=1.5, C2H2 1.81222e-10 percentage points at 3000 K/1 bar/C/O=1.5, C1H1N1_hcn 5.84907e-11 percentage points at 3000 K/1 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute N-allocation changes are C4N2 1.44718e-09 percentage points at 3000 K/1 bar/C/O=1.5, N2 6.97903e-10 percentage points at 3000 K/1 bar/C/O=1.5, C1H1N1_hcn 6.35592e-10 percentage points at 3000 K/1 bar/C/O=1.5. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 2.56e-09 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.31e-13, and formation residual 1.25e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 6.74e-09 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 2/75, ions nasa 2/75, neutral retained 2/75, neutral nasa 2/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

A bounded corrected-energy update using the explicitly selected NASA physical anchor is supported by the documented combustion correction. Prefer retaining thermals for that step; the second full NASA model is a sensitivity. This trace-species update has low immediate priority on the tested grid.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1021/j100806a601 <https://doi.org/10.1021/j100806a601>`_.
* `DOI 10.1021/j100840a514 <https://doi.org/10.1021/j100840a514>`_.
* `DOI 10.1021/j100352a010 <https://doi.org/10.1021/j100352a010>`_.
* `DOI 10.1021/je60041a030 <https://doi.org/10.1021/je60041a030>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/c4n2_candidate_test.py

The definition is ``tests/data/validation/c4n2_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

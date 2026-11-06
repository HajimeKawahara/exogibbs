CNC bounded candidates and fixed-network substitution
=====================================================

Recommend the evaluated energy as a bounded correction, retaining the present thermal function and its residual. The separate NASA result quantifies remaining thermal sensitivity; a complete Renner-Teller thermal update is deferred and full physical accuracy remains unknown. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral linear C-N-C isocyanomethylidyne, X2Pi-g, C2N1; exact NASA CNC, never CCN.

The ATcT ground-atom cycle gives D0=1221.195 kJ/mol, -120.167 relative to the old JANAF estimate (quoted 126 kJ/mol uncertainty). The 2020 erratum changes the 2017 experimental Hf298 result to 678+/-12 kJ/mol; that corrected value, not676, is compatible with the modern energy direction.

Original physical D0=1341.362000000 and selected D0=1221.195000000 kJ/mol. The retained G/H shift is +120.167000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

JANAF already includes 0/26.41 cm^-1 components of weight 2 each, so no additional Pi or doublet factor is added. Estimated stretches and strong Renner-Teller coupling remain unresolved. A NASA reference-model contrast is not a complete coupled-state reconstruction.

The original :doc:`cnc_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

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
     - 0.0875152707
   * - ions/retained
     - 6.53875473e-05 / 0.0269758746
     - 0.000708486128
   * - ions/nasa
     - 7.81423779e-05 / 0.0302711417
     - 0.000814890403
   * - neutral/current
     - 1 / 1
     - 0.0875166707
   * - neutral/retained
     - 6.53875473e-05 / 0.0269758749
     - 0.000708497475
   * - neutral/nasa
     - 7.81423779e-05 / 0.0302711422
     - 0.000814903455

At the ions current maximum-N-share condition (3000 K, 0.01 bar, C/O=1.5), the same-condition percentages are 0.0875152707 / 0.000708486128 / 0.000814890403 in order current, retained, nasa. The table above allows each model its own maximum.

At the neutral current maximum-N-share condition (3000 K, 0.01 bar, C/O=1.5), the same-condition percentages are 0.0875166707 / 0.000708497475 / 0.000814903455 in order current, retained, nasa. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C2N1 0.0159769 percentage points at 3000 K/0.01 bar/C/O=1.5, C1 0.00731808 percentage points at 3000 K/0.01 bar/C/O=1.5, C3H1 0.00280713 percentage points at 2000 K/1e-06 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/retained, the largest absolute N-allocation changes are C2N1 0.0868068 percentage points at 3000 K/0.01 bar/C/O=1.5, N2 0.0639738 percentage points at 2500 K/0.0001 bar/C/O=1.5, C1H1N1_hcn 0.0200062 percentage points at 3000 K/0.01 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute C-allocation changes are C2N1 0.0159573 percentage points at 3000 K/0.01 bar/C/O=1.5, C1 0.00730911 percentage points at 3000 K/0.01 bar/C/O=1.5, C3H1 0.00280676 percentage points at 2000 K/1e-06 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute N-allocation changes are C2N1 0.0867004 percentage points at 3000 K/0.01 bar/C/O=1.5, N2 0.0639411 percentage points at 2500 K/0.0001 bar/C/O=1.5, C1H1N1_hcn 0.0199817 percentage points at 3000 K/0.01 bar/C/O=1.5. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 1.51e-09 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.59e-06, H/C/O closure 2.51e-13, and formation residual 1.53e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.46e-07 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 53/75, ions nasa 53/75, neutral retained 53/75, neutral nasa 53/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend the evaluated energy as a bounded correction, retaining the present thermal function and its residual. The separate NASA result quantifies remaining thermal sensitivity; a complete Renner-Teller thermal update is deferred and full physical accuracy remains unknown.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1139/p66-029 <https://doi.org/10.1139/p66-029>`_.
* `DOI 10.1063/1.4978336 <https://doi.org/10.1063/1.4978336>`_.
* `DOI 10.1063/5.0010035 <https://doi.org/10.1063/5.0010035>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/cnc_candidate_test.py

The definition is ``tests/data/validation/cnc_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

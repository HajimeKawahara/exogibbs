P4S3 bounded candidates and fixed-network substitution
======================================================

If updating this estimated source, prefer the 2019 cycle-based H298 energy while retaining explicit thermal limitations. This is scientifically supported but very low priority for the tested hot gas. A tiny bulk response is not evidence of general irrelevance; a full cage thermal model remains deferred. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral P4S3 ideal gas; the old estimated table does not resolve cage isomers and has no molecular zero-K anchor.

The preferred 2019 reaction cycle gives Hf298=-94.5 kJ/mol versus JANAF -151.042, so G/H shift by+56.542 kJ/mol. White-P and rhombic-S elemental heat references are explicitly compatible. No D0 is invented from the constant-Cp extrapolation.

The physical comparison is H298-based: G/H shift by +56.542000 kJ/mol; S/Cp and the complete fit residual are unchanged. Neither a molecular H0 nor a physical D0 is available.

The less-converged direct-atomization route Hf298=-103.9 is a selected sensitivity, not the primary value. Its TAE0+/-6.4 kJ/mol uncertainty does not belong to the preferred cycle. Old constant-Cp thermals and absent cage frequency data remain limitations; the finite grid has an extremely small P4S3 inventory.

The original :doc:`p4s3_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

Calculation scope and counts
----------------------------

All variants independently start from named ``curated-ch2-v1`` after HCP/C2H2/CH2 adoption. Each uses 75 conditions in both 537-species ion and 421-species neutral networks. The 150 common current endpoints are reused once with file hashes, named factory potentials, species, formula matrix and elemental budgets verified.

Temperatures are 1500/2000/2500/3000/4000 K,  pressures 1e-6/1e-4/0.01/1/100 bar,  and C/O=0.5888436553555908/1/1.5. Only the explicitly declared target species change; the whole fixed gas network re-equilibrates.

There are **150 fresh main endpoints**, 12 tighter cold solves, 10 path solves, and 4 selected-alternative solves. FastChem executes 4 times at two ion-network conditions, including current and every main variant. These are actual executions, not that many distinct conditions. No neutral FastChem run is implied.

Selected alternatives are ``atomization_route_H298``. They cover two selected conditions per network, not a full-grid envelope or confidence interval.

Finite composition response
---------------------------

.. list-table:: Solved target response; shares are maxima on each model grid
   :header-rows: 1

   * - Network/model
     - Minimum/maximum x over current
     - Maximum P share (%)
   * - ions/current
     - 1 / 1
     - 6.33184875e-23
   * - ions/retained
     - 0.0107416526 / 0.182663453
     - 6.80145193e-25
   * - neutral/current
     - 1 / 1
     - 6.33185046e-23
   * - neutral/retained
     - 0.0107416526 / 0.182663453
     - 6.80145377e-25

At the ions current maximum-P-share condition (1500 K, 1 bar, C/O=0.588843655), the same-condition percentages are 6.33184875e-23 / 6.80145193e-25 in order current, retained. The table above allows each model its own maximum.

At the neutral current maximum-P-share condition (1500 K, 1 bar, C/O=0.588843655), the same-condition percentages are 6.33185046e-23 / 6.80145377e-25 in order current, retained. The table above allows each model its own maximum.

For ions/retained, the largest absolute P-allocation changes are P2 1.08302e-11 percentage points at 1500 K/0.0001 bar/C/O=1, C1H1P1 1.04632e-11 percentage points at 1500 K/1e-06 bar/C/O=1, P1 7.24976e-12 percentage points at 1500 K/1e-06 bar/C/O=1. Each entry has its own maximizing condition.

For ions/retained, the largest absolute S-allocation changes are C1S1 2.78824e-11 percentage points at 2000 K/1e-06 bar/C/O=1, S1 2.63012e-11 percentage points at 2000 K/1e-06 bar/C/O=1, H2S1 3.17524e-12 percentage points at 1500 K/0.01 bar/C/O=0.588844. Each entry has its own maximizing condition.

The selected ions/atomization_route_H298 control gives target x/current 0.022824527 to 0.242326366 at its two recorded conditions. These are sampled controls, not extrema of a full-grid uncertainty envelope.

The selected neutral/atomization_route_H298 control gives target x/current 0.022824527 to 0.242326366 at its two recorded conditions. These are sampled controls, not extrema of a full-grid uncertainty envelope.

The larger apparent changes in bulk P/S carriers above are numerical re-solve differences at this extremely small P4S3 inventory; they are not resolved physical redistribution driven by P4S3.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 1.4e-09 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.42e-06, H/C/O closure 2.23e-13, and formation residual 1.82e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.09e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 0/75, neutral retained 0/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

If updating this estimated source, prefer the 2019 cycle-based H298 energy while retaining explicit thermal limitations. This is scientifically supported but very low priority for the tested hot gas. A tiny bulk response is not evidence of general irrelevance; a full cage thermal model remains deferred.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1007/s11224-019-01352-7 <https://doi.org/10.1007/s11224-019-01352-7>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/p4s3_candidate_test.py

The definition is ``tests/data/validation/p4s3_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

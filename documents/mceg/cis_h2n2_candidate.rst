cis-H2N2 bounded candidates and fixed-network substitution
==========================================================

Recommend the exact-isomer retained-thermal ATcT energy change. Do not replace cis with a NASA isomer-mixture function; a separate cis-resolved thermal model can be assessed later. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral single cis-HNNH (Z-diazene), H2N2; neither trans nor H2NN nor their equilibrium mixture.

The exact-cis ATcT 1.228 cycle decreases D0 by9.259 kJ/mol. The benchmark isomer calculations contribute to the network and are not independent replication. This candidate retains cis-specific thermal and fit terms.

Original physical D0=1153.717000000 and selected D0=1144.458000000 kJ/mol. The retained G/H shift is +9.259000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

NASA N2H2 combines trans/cis/1,1 isomers and is expressly excluded. Energy alignment cannot make that mixture a single-cis species. Resonant anharmonicity and high-temperature cis-resolved states remain separate thermal work.

The original :doc:`cis_h2n2_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

Calculation scope and counts
----------------------------

All variants independently start from named ``curated-ch2-v1`` after HCP/C2H2/CH2 adoption. Each uses 75 conditions in both 537-species ion and 421-species neutral networks. The 150 common current endpoints are reused once with file hashes, named factory potentials, species, formula matrix and elemental budgets verified.

Temperatures are 1500/2000/2500/3000/4000 K,  pressures 1e-6/1e-4/0.01/1/100 bar,  and C/O=0.5888436553555908/1/1.5. Only the explicitly declared target species change; the whole fixed gas network re-equilibrates.

There are **150 fresh main endpoints**, 12 tighter cold solves, 10 path solves, and 0 selected-alternative solves. FastChem executes 4 times at two ion-network conditions, including current and every main variant. These are actual executions, not that many distinct conditions. No neutral FastChem run is implied.

Finite composition response
---------------------------

.. list-table:: Solved target response; shares are maxima on each model grid
   :header-rows: 1

   * - Network/model
     - Minimum/maximum x over current
     - Maximum N share (%)
   * - ions/current
     - 1 / 1
     - 1.29158381e-05
   * - ions/retained
     - 0.475969664 / 0.75699366
     - 9.77720752e-06
   * - neutral/current
     - 1 / 1
     - 1.29159051e-05
   * - neutral/retained
     - 0.475969664 / 0.75699366
     - 9.77725825e-06

At the ions current maximum-N-share condition (4000 K, 100 bar, C/O=0.588843655), the same-condition percentages are 1.29158381e-05 / 9.77720752e-06 in order current, retained. The table above allows each model its own maximum.

At the neutral current maximum-N-share condition (4000 K, 100 bar, C/O=0.588843655), the same-condition percentages are 1.29159051e-05 / 9.77725825e-06 in order current, retained. The table above allows each model its own maximum.

For ions/retained, the largest absolute N-allocation changes are H2N2 3.13863e-06 percentage points at 4000 K/100 bar/C/O=0.588844, N2 3.01628e-06 percentage points at 4000 K/100 bar/C/O=0.588844, C1H1N1_hcn 3.83302e-07 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 6.98e-10 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.23e-13, and formation residual 1.65e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.44e-06 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 20/75, neutral retained 20/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend the exact-isomer retained-thermal ATcT energy change. Do not replace cis with a NASA isomer-mixture function; a separate cis-resolved thermal model can be assessed later.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1080/002689799165530 <https://doi.org/10.1080/002689799165530>`_.
* `DOI 10.2478/acs-2024-0008 <https://doi.org/10.2478/acs-2024-0008>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/cis_h2n2_candidate_test.py

The definition is ``tests/data/validation/cis_h2n2_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

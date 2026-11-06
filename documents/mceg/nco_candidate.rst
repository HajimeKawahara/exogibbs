NCO bounded candidates and fixed-network substitution
=====================================================

Recommend the retained-thermal ATcT energy update, coordinating its production selection with HNCO. Separate and joint fixed-network variants quantify coupled response; they do not provide reaction covariance or validate an entire rovibronic model. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral linear N-C-O radical, X2Pi, C1N1O1; not anion, CON/CNO or HNCO.

The selected ATcT 1.228 physical D0=1301.838 kJ/mol is +31.984 above JANAF. HNCO-to-NCO and direct NCO threshold evidence participates in the evaluated network. The joint variant also changes HNCO by its own -17.260 kJ/mol G/H shift.

Original physical D0=1269.854000000 and selected D0=1301.838000000 kJ/mol. The retained G/H shift is -31.984000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

JANAF already counts the 0/95.6 cm^-1 spin-orbit components with weights 2 each; no extra orbital or spin factor is added. A complete rovibronic update remains separate from the few-percent aligned thermal contrast. The jointly changed HNCO->H+NCO reaction shifts G/H by -14.724 kJ/mol, preserving the matched energy ledger.

The original :doc:`nco_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

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
     - 1.79316758e-05
   * - ions/retained
     - 2.61613372 / 12.9945149
     - 4.69116616e-05
   * - ions/joint_hnco_nco
     - 2.61613212 / 12.9945149
     - 4.6911633e-05
   * - neutral/current
     - 1 / 1
     - 1.79319613e-05
   * - neutral/retained
     - 2.61613372 / 12.9945149
     - 4.69124086e-05
   * - neutral/joint_hnco_nco
     - 2.61613212 / 12.9945149
     - 4.691238e-05

At the ions current maximum-N-share condition (4000 K, 100 bar, C/O=1), the same-condition percentages are 1.79316758e-05 / 4.69116616e-05 / 4.6911633e-05 in order current, retained, joint_hnco_nco. The table above allows each model its own maximum.

At the neutral current maximum-N-share condition (4000 K, 100 bar, C/O=1), the same-condition percentages are 1.79319613e-05 / 4.69124086e-05 / 4.691238e-05 in order current, retained, joint_hnco_nco. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C1N1O1 4.66823e-06 percentage points at 4000 K/100 bar/C/O=0.588844, C1O1 4.56361e-06 percentage points at 4000 K/100 bar/C/O=0.588844, C1H1N1_hcn 5.99325e-07 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/retained, the largest absolute N-allocation changes are C1N1O1 2.898e-05 percentage points at 4000 K/100 bar/C/O=1, N2 2.53653e-05 percentage points at 4000 K/100 bar/C/O=1, C1H1N1_hcn 6.51259e-06 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/retained, the largest absolute O-allocation changes are C1N1O1 4.00035e-06 percentage points at 4000 K/100 bar/C/O=1, C1O1 3.65905e-06 percentage points at 4000 K/100 bar/C/O=1, H2O1 2.71766e-07 percentage points at 4000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

For ions/joint_hnco_nco, the largest absolute C-allocation changes are C1H1N1O1 1.98802e-05 percentage points at 2500 K/100 bar/C/O=0.588844, C1O1 1.86268e-05 percentage points at 2500 K/100 bar/C/O=0.588844, C1N1O1 4.66823e-06 percentage points at 4000 K/100 bar/C/O=0.588844. Each entry has its own maximizing condition.

For ions/joint_hnco_nco, the largest absolute N-allocation changes are C1H1N1O1 0.000123691 percentage points at 2500 K/100 bar/C/O=1.5, N2 0.000113579 percentage points at 2500 K/100 bar/C/O=1, C1N1O1 2.898e-05 percentage points at 4000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

For ions/joint_hnco_nco, the largest absolute O-allocation changes are C1H1N1O1 1.70742e-05 percentage points at 2500 K/100 bar/C/O=1.5, C1O1 1.60304e-05 percentage points at 2500 K/100 bar/C/O=1.5, C1N1O1 4.00035e-06 percentage points at 4000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

The joint variant reconstructs HNCO from copied primary-source bytes under this packet, without importing a new HNCO candidate. Its HNCO->H+NCO energy change is -14.724 kJ/mol. Compare this coupled response with the separate HNCO-only evaluation; do not combine independently normalized Hf values without the matched atomic ledger.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 9.31e-10 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.44e-06, H/C/O closure 2.34e-13, and formation residual 1.25e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.23e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 40/75, ions joint_hnco_nco 51/75, neutral retained 40/75, neutral joint_hnco_nco 51/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend the retained-thermal ATcT energy update, coordinating its production selection with HNCO. Separate and joint fixed-network variants quantify coupled response; they do not provide reaction covariance or validate an entire rovibronic model.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1063/1.1707013 <https://doi.org/10.1063/1.1707013>`_.
* `DOI 10.1063/1.472665 <https://doi.org/10.1063/1.472665>`_.
* `DOI 10.1063/1.463989 <https://doi.org/10.1063/1.463989>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/nco_candidate_test.py

The definition is ``tests/data/validation/nco_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

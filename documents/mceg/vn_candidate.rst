VN bounded candidates and fixed-network substitution
====================================================

Recommend the measured neutral-D0 update with retained thermals. The narrow threshold width is separately propagated at selected conditions; no cation energy, well depth or second zero-point correction is used. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral diatomic VN, N1V1, X3Delta, relative to neutral ground-state V+N; not VN+ or solid nitride.

The reported neutral threshold D0=4.9968(20) eV gives 482.117907554 kJ/mol, +22.269907554 above physical JANAF 459.848. The independent 2023 s-ccCA value 481.89 kJ/mol supports the central choice. The 2026 cation cycle does not supply a replacement neutral value.

Original physical D0=459.848000000 and selected D0=482.117907554 kJ/mol. The retained G/H shift is -22.269907554 kJ/mol; S/Cp and the complete fit residual are unchanged.

The quoted0.0020 eV width is tested at selected conditions. It does not cover inherited thermal or atomic-reference uncertainty. The 2025 single-potential De=48000 cm^-1 and 1-atm convention are not substituted for D0 or a complete1-bar thermal model.

The original :doc:`vn_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

Calculation scope and counts
----------------------------

All variants independently start from named ``curated-ch2-v1`` after HCP/C2H2/CH2 adoption. Each uses 75 conditions in both 537-species ion and 421-species neutral networks. The 150 common current endpoints are reused once with file hashes, named factory potentials, species, formula matrix and elemental budgets verified.

Temperatures are 1500/2000/2500/3000/4000 K,  pressures 1e-6/1e-4/0.01/1/100 bar,  and C/O=0.5888436553555908/1/1.5. Only the explicitly declared target species change; the whole fixed gas network re-equilibrates.

There are **150 fresh main endpoints**, 12 tighter cold solves, 10 path solves, and 8 selected-alternative solves. FastChem executes 4 times at two ion-network conditions, including current and every main variant. These are actual executions, not that many distinct conditions. No neutral FastChem run is implied.

Selected alternatives are ``lower_measured_D0``, ``upper_measured_D0``. They cover two selected conditions per network, not a full-grid envelope or confidence interval.

Finite composition response
---------------------------

.. list-table:: Solved target response; shares are maxima on each model grid
   :header-rows: 1

   * - Network/model
     - Minimum/maximum x over current
     - Maximum V share (%)
   * - ions/current
     - 1 / 1
     - 0.0110676339
   * - ions/retained
     - 1.9532897 / 5.96337234
     - 0.0314629462
   * - neutral/current
     - 1 / 1
     - 0.0110686752
   * - neutral/retained
     - 1.95328743 / 5.96337234
     - 0.0314630551

At the ions current maximum-V-share condition (3000 K, 100 bar, C/O=0.588843655), the same-condition percentages are 0.0110676339 / 0.027022868 in order current, retained. The table above allows each model its own maximum.

At the neutral current maximum-V-share condition (3000 K, 100 bar, C/O=0.588843655), the same-condition percentages are 0.0110686752 / 0.0270254098 in order current, retained. The table above allows each model its own maximum.

For ions/retained, the largest absolute V-allocation changes are N1V1 0.0206836 percentage points at 2500 K/100 bar/C/O=1, V1 0.0196667 percentage points at 2500 K/100 bar/C/O=1, O1V1 0.00744048 percentage points at 2000 K/100 bar/C/O=1. Each entry has its own maximizing condition.

For ions/retained, the largest absolute N-allocation changes are N1V1 2.43011e-06 percentage points at 2500 K/100 bar/C/O=1, N2 2.25928e-06 percentage points at 2500 K/100 bar/C/O=1, C1H1N1_hcn 3.66971e-07 percentage points at 3000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

The selected ions/lower_measured_D0 control gives target x/current 1.94199129 to 2.89184322 at its two recorded conditions. These are sampled controls, not extrema of a full-grid uncertainty envelope.

The selected ions/upper_measured_D0 control gives target x/current 1.96465383 to 2.9460211 at its two recorded conditions. These are sampled controls, not extrema of a full-grid uncertainty envelope.

The selected neutral/lower_measured_D0 control gives target x/current 1.94198906 to 2.89184322 at its two recorded conditions. These are sampled controls, not extrema of a full-grid uncertainty envelope.

The selected neutral/upper_measured_D0 control gives target x/current 1.96465152 to 2.94602109 at its two recorded conditions. These are sampled controls, not extrema of a full-grid uncertainty envelope.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 2.91e-10 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.29e-13, and formation residual 1.15e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.78e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 58/75, neutral retained 58/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend the measured neutral-D0 update with retained thermals. The narrow threshold width is separately propagated at selected conditions; no cation energy, well depth or second zero-point correction is used.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1063/1.4953782 <https://doi.org/10.1063/1.4953782>`_.
* `DOI 10.1080/00268976.2023.2248292 <https://doi.org/10.1080/00268976.2023.2248292>`_.
* `DOI 10.1063/5.0349800 <https://doi.org/10.1063/5.0349800>`_.
* `DOI 10.1038/s41598-025-17514-z <https://doi.org/10.1038/s41598-025-17514-z>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/vn_candidate_test.py

The definition is ``tests/data/validation/vn_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

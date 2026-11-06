NH2 bounded candidates and fixed-network substitution
=====================================================

Recommend the retained-thermal ATcT energy update for a separately authorized bounded adoption. Keep the NASA thermal candidate as a measured model contrast; full-range superiority is not established. The 2026 spin-convention issue does not block this energy decision. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral bent amidogen, X2B1, H2N1; neither an anion nor ammonia.

The same-version ATcT 1.228 balance corrects the JANAF physical D0 by +4.089 kJ/mol. The IUPAC radical assessment supports the direction of lower molecular formation enthalpy. NASA carries a different molecular/atomic thermal model, so its additional effect is evaluated separately.

Original physical D0=709.647000000 and selected D0=713.736000000 kJ/mol. The retained G/H shift is -4.089000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

The inspected 2026 RRHO source reproduces its printed Q/S with ground electronic weight 1 and leaves electron-spin normalization unresolved. Its entropy is about R ln2 below evaluated alternatives. It is not used here; no arbitrary factor 2 or new rovibrational model is inserted.

The original :doc:`nh2_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

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
     - 1.22585455
   * - ions/retained
     - 1.12987558 / 1.38799803
     - 1.38507341
   * - ions/nasa
     - 1.33402237 / 1.49878445
     - 1.63534844
   * - neutral/current
     - 1 / 1
     - 1.22585778
   * - neutral/retained
     - 1.12987558 / 1.38799803
     - 1.38507706
   * - neutral/nasa
     - 1.33402238 / 1.49878445
     - 1.63535275

At the ions current maximum-N-share condition (4000 K, 100 bar, C/O=0.588843655), the same-condition percentages are 1.22585455 / 1.38507341 / 1.63534844 in order current, retained, nasa. The table above allows each model its own maximum.

At the neutral current maximum-N-share condition (4000 K, 100 bar, C/O=0.588843655), the same-condition percentages are 1.22585778 / 1.38507706 / 1.63535275 in order current, retained, nasa. The table above allows each model its own maximum.

For ions/retained, the largest absolute N-allocation changes are H2N1 0.159219 percentage points at 4000 K/100 bar/C/O=0.588844, N2 0.15399 percentage points at 4000 K/100 bar/C/O=0.588844, C1H1N1_hcn 0.0309348 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/nasa, the largest absolute N-allocation changes are H2N1 0.409494 percentage points at 4000 K/100 bar/C/O=0.588844, N2 0.396038 percentage points at 4000 K/100 bar/C/O=0.588844, C1H1N1_hcn 0.0796024 percentage points at 4000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 1.31e-09 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.29e-13, and formation residual 1.53e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.23e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 63/75, ions nasa 69/75, neutral retained 63/75, neutral nasa 69/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend the retained-thermal ATcT energy update for a separately authorized bounded adoption. Keep the NASA thermal candidate as a measured model contrast; full-range superiority is not established. The 2026 spin-convention issue does not block this energy decision.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1063/1.1724828 <https://doi.org/10.1063/1.1724828>`_.
* `DOI 10.5281/zenodo.20662187 <https://doi.org/10.5281/zenodo.20662187>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/nh2_candidate_test.py

The definition is ``tests/data/validation/nh2_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

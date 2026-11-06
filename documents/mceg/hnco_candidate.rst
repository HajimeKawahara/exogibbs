HNCO bounded candidates and fixed-network substitution
======================================================

Recommend retained-thermal ATcT energy adoption, with HNCO/NCO bookkeeping reviewed jointly before production selection. A low-temperature partition table is not extrapolated into the hot candidate. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral singlet isocyanic acid H-N=C=O, C1H1N1O1; not HOCN, HCNO, HONC or cyanate.

ATcT 1.228 gives physical D0=1760.762 kJ/mol, +17.260 relative to the 1970 JANAF energy. Later dissociation thresholds and the thermochemical network support the revision. Source thermal differences after alignment are below about 0.7%, so a simultaneous thermal replacement is not required.

Original physical D0=1743.502000000 and selected D0=1760.762000000 kJ/mol. The retained G/H shift is -17.260000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

The inspected 2021 partition-function analysis ends at 500 K and cannot establish a new1500-4000 K function. This packet evaluates HNCO alone; the companion NCO packet independently copies the required HNCO source bytes and evaluates matched HNCO/NCO jointly, avoiding dependence on this new candidate packet.

The original :doc:`hnco_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

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
     - 0.000127266058
   * - ions/retained
     - 1.68030379 / 3.99056929
     - 0.000223524442
   * - neutral/current
     - 1 / 1
     - 0.000127268208
   * - neutral/retained
     - 1.68030379 / 3.99056929
     - 0.000223525764

At the ions current maximum-N-share condition (4000 K, 100 bar, C/O=1), the same-condition percentages are 0.000127266058 / 0.00021384564 in order current, retained. The table above allows each model its own maximum.

At the neutral current maximum-N-share condition (4000 K, 100 bar, C/O=1), the same-condition percentages are 0.000127268208 / 0.000213849253 in order current, retained. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C1H1N1O1 1.98802e-05 percentage points at 2500 K/100 bar/C/O=0.588844, C1O1 1.85711e-05 percentage points at 2500 K/100 bar/C/O=0.588844, C1H4 4.6543e-06 percentage points at 2000 K/100 bar/C/O=0.588844. Each entry has its own maximizing condition.

For ions/retained, the largest absolute N-allocation changes are C1H1N1O1 0.000123691 percentage points at 2500 K/100 bar/C/O=1.5, N2 0.000113239 percentage points at 2500 K/100 bar/C/O=1, C1H1N1_hcn 2.4478e-05 percentage points at 3000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

For ions/retained, the largest absolute O-allocation changes are C1H1N1O1 1.70742e-05 percentage points at 2500 K/100 bar/C/O=1.5, C1O1 1.59824e-05 percentage points at 2500 K/100 bar/C/O=1.5, H2O1 3.87961e-06 percentage points at 2000 K/100 bar/C/O=1.5. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 9.31e-10 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.4e-06, H/C/O closure 2.52e-13, and formation residual 1.82e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.05e-08 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 42/75, neutral retained 42/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend retained-thermal ATcT energy adoption, with HNCO/NCO bookkeeping reviewed jointly before production selection. A low-temperature partition table is not extrapolated into the hot candidate.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1063/1.1707013 <https://doi.org/10.1063/1.1707013>`_.
* `DOI 10.1063/1.472665 <https://doi.org/10.1063/1.472665>`_.
* `DOI 10.1093/mnras/stab1163 <https://doi.org/10.1093/mnras/stab1163>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/hnco_candidate_test.py

The definition is ``tests/data/validation/hnco_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

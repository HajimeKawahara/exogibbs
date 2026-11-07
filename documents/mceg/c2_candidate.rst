C2 bounded candidates and fixed-network substitution
====================================================

Recommend retained thermals with the ATcT energy. The measured ground-referenced dissociation supports the physical change independently of equilibrium fitting. A later eight-state thermal candidate should make dissociation and isotope conventions explicit. **Production is unchanged in this candidate review.**

Identity and scientific construction
------------------------------------

Neutral equilibrated carbon dimer with X1Sigma+g ground level and thermally accessible triplet manifolds; not a triplet-only species.

ATcT 1.228 gives D0=602.800 kJ/mol, compared with physical JANAF 593.109. The independent 2021 photofragment result 602.804(29) kJ/mol already corrects initial triplet energy and atomic fine structure and supports this energy choice.

Original physical D0=593.109000000 and selected D0=602.800000000 kJ/mol. The retained G/H shift is -9.691000000 kJ/mol; S/Cp and the complete fit residual are unchanged.

The original source packet preserves the 44189-level eight-state diagnostic. No extra electronic factor or symmetry divisor belongs in its explicit bosonic state sum. Full-distribution/bound-level and missing-manifold questions remain separate; its much smaller thermal difference does not require changing thermals with this energy update.

The original :doc:`c2_source_review` and its packet/receipt are preserved as the earlier stage. No energy correction is inferred from the fitted 1/T intercept. No extra ZPE, electronic factor, nuclear-spin multiplier or symmetry division is added.

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
     - Maximum C share (%)
   * - ions/current
     - 1 / 1
     - 0.772762364
   * - ions/retained
     - 1.33096754 / 2.17502377
     - 1.20695891
   * - neutral/current
     - 1 / 1
     - 0.772764388
   * - neutral/retained
     - 1.33096739 / 2.17502377
     - 1.20696204

At the ions current maximum-C-share condition (2500 K, 0.0001 bar, C/O=1.5), the same-condition percentages are 0.772762364 / 1.20695891 in order current, retained. The table above allows each model its own maximum.

At the neutral current maximum-C-share condition (2500 K, 0.0001 bar, C/O=1.5), the same-condition percentages are 0.772764388 / 1.20696204 in order current, retained. The table above allows each model its own maximum.

For ions/retained, the largest absolute C-allocation changes are C2 0.434197 percentage points at 2500 K/0.0001 bar/C/O=1.5, C1 0.20702 percentage points at 2500 K/0.0001 bar/C/O=1.5, C3H1 0.0690129 percentage points at 2500 K/0.0001 bar/C/O=1.5. Each entry has its own maximizing condition.

Every species carrying a tracked element has its largest allocation response archived in ``summary.json``. Equilibrium-constant factors are not substituted for solved abundances. A finite hot hydrogen-rich inventory does not constrain photochemistry, transport, condensation or arbitrary atmospheric compositions.

Numerical evidence and decision
-------------------------------

Each main and selected potential passes 2752 scalar/JAX temperatures, including 251 off-grid points. Independent source expansion agrees in G/H within 3.49e-10 J/mol. All archived solves converge from cold starts. Main all-element relative closure is at most 1.47e-06, H/C/O closure 2.39e-13, and formation residual 1.17e-12 in ln units. Trace-element closure is stated separately from major elements. Independent FastChem agrees within 1.37e-07 dex.

Same-model feasibility-corrected Gibbs signs exceed the displayed floating-point diagnostic scale in ions retained 50/75, neutral retained 50/75. Remaining cases are unresolved. The Gibbs scale and the path monotonicity tolerance are finite numerical checks, not rigorous sign resolution for arbitrarily small responses, not source ranking, and not physical uncertainty.

Recommend retained thermals with the ATcT energy. The measured ground-referenced dissociation supports the physical change independently of equilibrium fitting. A later eight-state thermal candidate should make dissociation and isotope conventions explicit.

Complete physical/thermal uncertainty remains unknown. The executed candidate review is mergeable within that scope; unavailable historical full texts do not create an unlimited merge hold. Any subsequent production adoption needs its own explicit selection and packaged-data verification.

Sources and replay
------------------

The frozen source review records exact original identities, numerical tables, access levels and source ancestry. Useful primary references include:

* `DOI 10.1073/pnas.2113315118 <https://doi.org/10.1073/pnas.2113315118>`_.
* `DOI 10.1093/mnras/sty2050 <https://doi.org/10.1093/mnras/sty2050>`_.
* `DOI 10.1093/mnras/staa1954 <https://doi.org/10.1093/mnras/staa1954>`_.

Replay the numerical checks with::

   pytest tests/model_evidence/c2_candidate_test.py

The definition is ``tests/data/validation/c2_candidate/20261106/candidate.json``. Shared review-only evaluators are ``mceg/reviews/pending_candidate.py`` and ``pending_equilibrium_impact.py``; the source ledger is ``pending_molecule_inputs.py``. They do not modify production. MCEG separates source construction, finite numerical support and candidate direction from unknown physical accuracy and unresolved infinitesimal Gibbs signs.

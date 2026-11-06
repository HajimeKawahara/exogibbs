HCP retained-thermal energy candidates and finite substitution
==============================================================

Both modern HCP formation-energy alternatives substantially reduce HCP in the
tested gas networks. At 1500 K, 1 bar and C/O=1.5, its share of phosphorus falls
from 94.288% to 8.163% with the CCCBDB anchor or 3.384% with the 2001 theoretical
anchor. The numerical response supports correcting the old energy estimate;
it does not determine which modern central value is more accurate.
**Production remains unchanged.** This candidate extension follows the frozen
:doc:`hcp_source_review`; statements of no new solves in that original packet
describe the earlier source-review stage.

Identity, source energies and unchanged thermal model
-----------------------------------------------------

The target is neutral linear H-C-P gas, ``C1H1P1``, CAS 6829-52-3, ground
X 1Sigma+, not H-P-C or a condensed polymer. The source review inspected the
official JANAF C-056 gas table and printed page 604. Its December 1969 energy
estimate uses an HCN-like C-H bond and old CP thermochemistry; the quoted
uncertainty is +/-63 kJ/mol. No neutral HCP record exists in the pinned NASA
collection, so this review supplies no invented NASA ratio or analogue.

The official `CCCBDB HCP entry
<https://cccbdb.nist.gov/exp2x.asp?casno=6829523>`_ gives Hf0=213.50 +/-20.00
kJ/mol. It explicitly corrects the JANAF energy using modern CP thermochemistry;
it is an evaluated adjustment, not a direct independent HCP measurement.
The complete author-hosted `2001 CCSD(T) atomization study
<https://doi.org/10.1039/B106927F>`_ gives Hf0=225 +/-8 kJ/mol in its HCP paragraph.
The latter number uses the body-specific uncertainty, not the abstract's broader
+/-10 wording. These source widths are sensitivity inputs, not Gaussian
confidence intervals or a complete correlated uncertainty budget.

.. list-table:: Independent alternatives on the same JANAF atomic reference
   :header-rows: 1

   * - Model
     - Hf0 (kJ/mol)
     - Common-reference D0 (kJ/mol)
     - Change in G and H (kJ/mol)
   * - Current JANAF anchor
     - 151.192
     - 1091.582
     - 0
   * - CCCBDB
     - 213.500 +/-20
     - 1029.274
     - +62.308
   * - 2001 theory
     - 225.000 +/-8
     - 1017.774
     - +73.808

Both alternatives retain the current gas atomic sources. The official
`CCCBDB reference-state table <https://cccbdb.nist.gov/refstatex.asp>`_
specifies white crystalline P with H298-H0=5.360 kJ/mol. Independently, JANAF's
P gas atom implies 6.197-(316.390-315.553)=5.360 kJ/mol. Therefore no additional
white/red phosphorus phase correction is applied. The old printed narrative's
40 kcal/mol and selected table's 149.900 kJ/mol differ by the historical
17.460 kJ/mol phosphorus conversion; the selected table is the baseline.

The 2001 paper's rounded atomic Hf0 values, C=711, H=216 and P=316 kJ/mol,
give its own total atomization energy 1018 kJ/mol. This is a separate reference
control, 0.226 kJ/mol above the common-JANAF result; it is not silently applied
as another correction. Neither total D0 is the paper's single H-CP bond De,
an H298 anchor, or the empirical fit intercept R*a. The fit is never evaluated
at zero temperature.

For either alternative, G_candidate(T)=G_current(T)+delta-Hf0 and
H_candidate(T)=H_current(T)+delta-Hf0. S and Cp are unchanged. The literal
five-coefficient FastChem function, its source-fit residual, 1-bar convention,
and existing thermal/isotope/electronic assumptions are retained. This is
consistent with an energy-only update: it does not newly validate those older
assumptions. Both candidates are independent substitutions into
``curated-tio2-v1``; the second is not applied on top of the first.

The 2024 HCP work, `DOI 10.1021/acs.jpca.4c01823
<https://doi.org/10.1021/acs.jpca.4c01823>`_, reports ground-state rovibrational
data intended through 3000 K. The source review actually inspected its public
supplement: three PES/DMS Fortran routines, without the required states or Q
moments. A PES alone cannot supply the requested thermodynamic replacement.
There is no 2024 thermal candidate or unsupported extension to 4000 K here.
That data gap is separate from the completed energy-only calculations.

Calculation scope and distinct counts
-------------------------------------

The five temperatures are 1500, 2000, 2500, 3000 and 4000 K; the five pressures
are 1e-6, 1e-4, 0.01, 1 and 100 bar; C/O is 0.5888436553555908 (the retained
elemental baseline), 1 or 1.5. Each 75-condition
grid uses either 537 species including ions or 421 neutral species. The species,
elements, formula matrix, budgets and potentials are checked against the actual
named production factory before reusing the archived baseline.

* 150 common baseline endpoints are reused and hash-verified, once per network.
* 300 fresh endpoints are solved: two anchors times two networks times 75 cases.
* 12 fresh tighter cold controls cover current and both central anchors at two
  selected conditions in each network.
* 16 fresh quoted-width edge controls cover both signs of each source width at
  those two conditions in each network. These are not full-grid uncertainty runs.
* Four standalone FastChem executions give two current/candidate comparisons
  at one distinct condition, 1500 K, 1 bar, C/O=1.5. The current calculation is
  deliberately executed twice; these are not four distinct conditions.

All 328 fresh ExoGibbs solves converge from cold starts. The primary tolerance
is 5e-14 and the selected controls use 2e-14. No source, production preset,
packaged dataset or earlier receipt is modified.

Phosphorus redistribution and sensitivities
-------------------------------------------

.. list-table:: Percent of phosphorus at 1500 K, 1 bar, C/O=1.5 (ion network)
   :header-rows: 1

   * - Carrier
     - Current
     - CCCBDB
     - 2001 theory
   * - HCP
     - 94.2882
     - 8.1631
     - 3.3837
   * - PH2
     - 3.3902
     - 43.3488
     - 45.1807
   * - PH3
     - 2.0838
     - 26.6451
     - 27.7712
   * - P2
     - 0.1248
     - 20.3977
     - 22.1581

The principal recipients at this point are PH2, PH3 and P2, not CP. Shares count
both phosphorus atoms in P2 and use the conserved elemental P budget. HCP mole
fractions change from 4.16069e-7 to 3.60215e-8 or 1.49314e-8. A large change in
the trace phosphorus inventory is not a comparable change in the bulk gas.

The current maximum-HCP-share condition is instead 1500 K, **0.01 bar**, C/O=1.5.
There HCP changes from 99.9638% to 87.0758% or 63.5395%; P2 changes from
0.000502% to 8.3264% or 28.0253%. These remain the maximum HCP shares on the
respective tested grids. Thus neither central candidate removes HCP dominance
at every condition. Across the ion grid, HCP mole-fraction ratios span
0.00676527-0.87107328 (CCCBDB) and 0.00269046-0.63562517 (theory).
The neutral grid gives the same conclusion, with maximum HCP shares differing
by less than 0.000001 percentage point. Numerical closeness of the two networks
does not assess omitted condensates or disequilibrium chemistry.

.. list-table:: HCP phosphorus share at source-width edges (percent, ion network)
   :header-rows: 1

   * - Source width
     - 0.01 bar, lower/upper energy
     - 1 bar, lower/upper energy
   * - CCCBDB +/-20 kJ/mol
     - 98.5220 / 41.7549
     - 31.7075 / 1.7353
   * - 2001 theory +/-8 kJ/mol
     - 81.5126 / 42.9626
     - 6.2694 / 1.8052

Both use 1500 K and C/O=1.5. Increasing Hf0 reduces HCP in these controls, and
both ends of either quoted width reduce HCP relative to the current result in
both networks. The direction is robust within these selected energy controls;
whether HCP remains the majority phosphorus carrier at 0.01 bar is not.
The ranges are not a confidence interval, full-grid envelope, or estimate of
inherited thermal and competing-species errors. No averaging of the two
sources or statistical combination of their widths is justified here.

Verification and scientific decision
------------------------------------

At 2501 one-kelvin nodes, the reconstructed current dimensionless potential
matches the actual production factory within 2.14e-14. Both independent
literal-fit checks and energy shifts agree within 2.85e-14. The G/H shift
and JAX derivative tests agree within 5.83e-10 J/mol; S/Cp are unchanged,
and source-fit residuals are preserved. Replay tests allow 1e-8 J/mol for
dimensional cancellation while explicitly rejecting a 1e-5 J/mol perturbation.
Dimensionless and source-value checks retain their stricter tolerances.

All species remain finite and positive without concentration floors. The main
endpoints have maximum relative P and H/C/N/O/P/S balance error 1.42e-10,
all-element error 1.49e-6, mass-action residual 1.44e-12 in ln units, and ion
charge residual 1.74e-14. Tighter controls change HCP ln mole fraction by less
than 8.89e-14. FastChem agrees within 7.91e-9 dex. All controls independently
pass their conservation, charge and mass-action assertions.

Feasibility-corrected same-model Gibbs exchange signs are resolved above the
displayed floating-point scale for 64/75 CCCBDB and 65/75 theory conditions in
each network. The remaining 42 of 300 candidate/condition comparisons are
explicitly unresolved, rather than rounded into a universal claim. This scale
is a numerical diagnostic, not a rigorous error bound. Absolute Gibbs values
from different energy models cannot decide which source is physically right.

The shared modern direction is scientifically preferable to retaining the old
central energy solely because it is already in production. For a separately
authorized bounded update, the **CCCBDB anchor is the recommended evaluated
primary option**, with the 2001 direct atomization result retained as an
independent sensitivity. CCCBDB explicitly corrects the identified CP energy
problem within the inherited JANAF construction and publishes a usable
uncertainty; it requires no replacement of the present thermal model. This is
a transparent source-selection convention, **not proof that 213.50 kJ/mol is
more accurate than 225 kJ/mol**. The theoretical calculation avoids the old
bond analogy and is valuable independent evidence, but its narrower quoted
width alone does not establish superiority. Central-value physical preference
between the two modern results remains unestablished; both overlap within
their reported widths. Complete physical uncertainty also remains unknown.

The evidence is sufficient to review this candidate calculation now. An
explicit adoption decision, retained competing-species limits, and any modern
thermal replacement belong to subsequent scoped work. The absent 2024 states
file does not block merging these energy calculations, nor does merging them
implicitly authorize adoption.

The continuous helper is ``mceg/reviews/hcp_candidate.py``; the finite runner is
``mceg/reviews/hcp_equilibrium_impact.py``. Frozen packets are under
``tests/data/validation/hcp_candidate/20261106`` and
``tests/data/validation/hcp_equilibrium_impact/20261106``. The contract
``mceg/contracts/hcp_candidate.yaml`` records source scope, numerical support,
scoped selection and remaining unknowns separately. Execution receipts preserve
the commands and input/output hashes from their actual runs; later replay tests
and this document are pinned by the contract, without rewriting those receipts.

Run the offline checks with::

   pytest tests/model_evidence/hcp_source_review_test.py \
          tests/model_evidence/hcp_candidate_test.py \
          tests/model_evidence/hcp_equilibrium_impact_test.py

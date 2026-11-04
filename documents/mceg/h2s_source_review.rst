H2S: retain the anchor, resolve high-temperature state completeness
===================================================================

The current H2S column remains **NASA thermal functions + ATcT 1.172 energy**.
The modern AYT2/TIPS partition sums provide a useful low-temperature check,
but their distributed high-temperature state coverage does not yet justify
replacing the production thermal model across 1500--4000 K. This is a scoped
retention decision, not a claim that the inherited NASA functions have complete
physical validation. No production coefficients or equilibrium cases change.

H2S is a high-priority sulfur reservoir. In the 75 archived
``curated-h2o-v1`` cases it holds up to 91.4008% of the sulfur, at 1500 K,
100 bar and C/O = 0.588844, with mole fraction 2.07029e-5.
The reported detection in HD 189733b motivates this review
(`Fu et al. 2024 <https://doi.org/10.1038/s41586-024-07760-y>`_), but these
gas-only equilibrium endpoints do not validate that planet, a colder
atmosphere, condensation, transport, photochemistry or spectral intensities.

Balanced energy and atomic reference
------------------------------------

The reaction is ``2 H(g) + S(g) -> H2S(g)`` at one bar. The frozen
`ATcT 1.172 table <https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php>`_
gives 0 K formation enthalpies of H = 216.034, S = 276.830 and
H2S = -17.370 kJ/mol. Thus the balanced atomization energy is
726.268 kJ/mol, versus native NASA 724.673796 and JANAF 724.391 kJ/mol.
This is ground-state D0; no potential-well De or extra zero-point correction
is inserted. The first fragmentation H2S -> H + SH instead costs
376.214 kJ/mol using the same ATcT version. Confusing that threshold with total
atomization would make a bound-state completeness check misleading.

The existing hybrid shift is 192.2353575 K in D0/k, giving
``K_curated / K_NASA = 1.13673135`` at 1500 K and 1.04923239 at 4000 K.
NASA's native gas constant is preserved in its dimensionless functions,
including its energy anchor. Dividing the raw energy difference by modern R
alone would miss this convention effect. All displayed caloric comparisons
use a common modern R. The entire unchanged production H2S reaction column
and all archived H2S potentials are reproduced.

NASA H2S cites Gurvich (1989), part 1 p298 and part 2 p181; its stored
H298-H0 is 9958.421 J/mol and natural molecular mass is 34.08088 u.
NASA S cites Cox (1989), Martin (1990) and Gordon (1999). S is referenced to
its lowest fine-structure level, 3P2, rather than the term centroid.
The `NIST S I table <https://physics.nist.gov/PhysRefData/Handbook/Tables/sulfurtable5.htm>`_
gives the 3P2/1/0 energies 0, 396.055 and 573.640 cm-1 with weights 5, 3 and 1.
The packet also compares the next 1D2 and 1S0 levels. NASA's inferred atomic
partition sum changes from 7.62958 at 1500 K to 8.59645 at 4000 K, so it is
neither a fixed ground degeneracy of 5 nor an unsplit constant 9.
H has electronic weight approximately 2 after nuclear-spin removal.
The aligned molecular comparison keeps these atomic thermal functions fixed;
native JANAF atomic effects are recorded separately.

The displayed ATcT uncertainties are not independent error bars: the network
covariance and isotope-resolved atomization energies have not been propagated.
The current same-version balanced energy is retained; this review does not
mix newer ATcT releases or replace only the sulfur atom's enthalpy.

Modern state sums and normalization
-----------------------------------

`AYT2 (Azzam et al. 2016) <https://doi.org/10.1093/mnras/stw1133>`_ supplies
220630 rovibrational states of 1H2-32S in its current distribution, with
J <= 40 and energies through 34636.207111 cm-1. The paper's partition-function
convergence figure extends to 2000 K. Its transition-frequency coverage and
lower-state cutoff concern spectral completeness and must not be substituted
for thermodynamic convergence criteria.

The current list contains measured energies from
`Chubb et al. 2018 <https://doi.org/10.1016/j.jqsrt.2018.07.012>`_.
It preserves the original calculated energy in a separate column, allowing
an exact comparison without obtaining a historical state file. The para ground
is zero; the lowest ortho state is 13.746310 cm-1 above it. The two manifolds
retain this common zero. They are assumed to equilibrate; a frozen ortho/para
ratio is outside the reviewed model.

Weights include the para/ortho factors 1 and 3. Divide the full sum by the
nuclear-spin product **4 exactly once** for thermochemistry. The allowed-state
list already implements exchange symmetry; do not divide by another symmetry
number of 2. No additional electronic degeneracy is needed for the ground
singlet molecular list. The representative isotope's internal states are
combined with the existing natural mass for the aligned comparison. A separate
33.98772106 u translation sensitivity is recorded; neither construction is a
complete isotope-resolved reaction or natural-isotope mixture calculation.

The current file contains one assignment inconsistency: state 76644 has
``J=13, gtot=99, symmetry=B2``. B2 requires nuclear-spin weight 3, so the
printed J would imply gtot=81; adjacent rows have J=16 and gtot=99.
The original line and neighboring lines are archived. The data do not identify
which field is wrong, so the original weight is preserved. Changing 99 to 81
as a labelled sensitivity changes Q by at most 1.70 ppm on the sampled grid.
MARVEL energy replacement also changes Q by less than 4.6 ppm. Neither
explains the large high-temperature difference.

The official definition currently says maximum temperature 1500 K and
maximum energy 10000 cm-1, whereas the actual state maximum is higher, the
published PF ends at 3000 K and TIPS ends at 4000 K. These describe different
or inconsistent source limits; none is silently promoted to a uniform
physical validity domain. The heat-capacity file actually contains one-K
nodes from 200 to 3000 K, despite the webpage's coarser-interval description.

High-temperature disagreement and the decision
----------------------------------------------

`TIPS2017 <https://doi.org/10.1016/j.jqsrt.2017.03.045>`_ Section 3.31
explicitly describes the main-isotope sum as AYT2-derived and converged to
4000 K. `TIPS2021 <https://doi.org/10.1016/j.jqsrt.2021.107713>`_ inherits
that calculation. `TIPS2025 <https://doi.org/10.1016/j.jqsrt.2025.109568>`_
updates constants and still lists 4000 K; its methods caution that some
high-temperature limits are optimistic. The 2025
`corrigendum <https://doi.org/10.1016/j.jqsrt.2025.109618>`_ concerns an NO2
isotopologue, not H2S. The packet's three TIPS arrays and direct state sum
agree within 6 ppm at the sampled nodes. This reflects shared state content
and does not independently establish the reported high-temperature convergence.

.. list-table:: Common natural mass, common ATcT anchor and NASA atoms
   :header-rows: 1

   * - T (K)
     - AYT2 / NASA Q and formation K
     - AYT2 Cp (J/mol/K)
     - NASA Cp (J/mol/K)
     - JANAF Cp (J/mol/K)
     - AYT2 minus NASA G (J/mol)
   * - 1500
     - 0.997749
     - 51.6838
     - 51.8429
     - 51.476
     - 28.11
   * - 2000
     - 0.995443
     - 54.2903
     - 55.2037
     - 54.656
     - 75.95
   * - 3000
     - 0.970756
     - 51.6456
     - 58.8371
     - 57.840
     - 740.32
   * - 4000
     - 0.891967
     - 43.3700
     - 61.0697
     - 59.509
     - 3802.23

G is measured from the same molecular H(0); the packet also contains H, S,
Q, Cp, formation ln(K), native JANAF differences, all intermediate temperature
nodes and exact energy moments. The calorics are derived analytically from
state energies, avoiding derivatives of rounded partition tables.
Published AYT2 Cp differs from the replay by roughly 0.002--0.003 J/mol/K;
constants or generation details are not established, so the residual is
preserved rather than normalized away.

At 4000 K, the J=40 boundary shell alone carries 0.8507% of distributed Q,
and J>=35 carries 7.0181%. Increasing high-temperature disagreement and the
falling finite-list Cp make missing-state convergence a material concern.
States above the first H+SH threshold contribute only 4.35e-8 of Q here;
that small fraction does not bound omitted high-J states. These diagnostics
**do not establish a rigorous 10.8% missing-state correction** or validate
NASA's exact value. They do show that direct adoption of the distributed
AYT2/TIPS table needs a new convergence assessment.

The June 1977 `JANAF H2S table <https://janaf.nist.gov/tables/H-080.html>`_
also differs thermally from NASA: after aligning D0 and atoms its formation K
is 0.980605 times NASA at 1500 K and 0.942689 at 4000 K. Its Cp stays much
closer to NASA than to the truncated AYT2 result at 4000 K, but the remaining
NASA/JANAF disagreement prevents a blanket physical-accuracy claim. NASA is
retained provisionally while a better-converged alternative is prepared.

Sulfur budget and concrete follow-up
------------------------------------

The saved grid contains 75 converged, 537-species gas-only endpoints at
1500, 2000, 2500, 3000 and 4000 K, five pressures from 1e-6 to 100 bar,
and C/O = 0.588844, 1 and 1.5. Sulfur conservation closes below 2.95e-12
relative error. The largest reservoir fractions occur in different cases:
H2S 0.914008, S 0.999976, SiS 0.999976, CS 0.934736, SH 0.480123,
S2 0.0188162, SO 0.00249201, COS 1.91891e-5 and SO2 4.86763e-6.
These are context and conservation checks on previously computed endpoints,
not a new substitution or an abundance response to AYT2.

The next useful work is **calculation**: extend J and vibrational coverage,
check convergence of Q and its first two energy moments through 4000 K,
and reconcile the TIPS2017 claim with the distributed state list. Then fit a
consistent thermal candidate at fixed balanced ATcT energy and perform finite
substitution with sulfur-budget checks. Resolve the isolated J/weight assignment
and source metadata before interpreting precision-level residuals.

The DOI papers above are already inspected, including the relevant TIPS2025
methods and correction; no additional paper is required to merge this scoped
source review. Gurvich's original pages and the older Haar thermodynamic
calculation would be useful for tracing the legacy NASA/JANAF difference,
but they are not prerequisites for documenting the present evidence. No DOI
is invented for an unverified legacy book/table.

Evidence and replay
-------------------

``mceg/contracts/h2s_source_review.yaml`` records separate supported claims for
source construction, normalized numerical comparison and scoped retention,
while physical accuracy remains ``unknown``. The compact packet in
``tests/data/validation/h2s_source_review/20261104`` includes original source
URLs, hashes, acquisition/inspection limits, complete extracted numerical
arrays and a receipt. Full papers are not redistributed.

.. code-block:: console

   PYTHONPATH=src python mceg/reviews/h2s_source_review.py
   pytest tests/model_evidence/h2s_source_review_test.py
   PYTHONPATH=/path/to/mceg/src python -m mceg check \
     mceg/contracts/h2s_source_review.yaml \
     --assessment scoped_retention_review --root . --json

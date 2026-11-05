Formaldehyde candidates and finite substitution
===============================================

Decision
--------

Prefer a bounded energy-only update of formaldehyde to the same-version ATcT
1.228 atomization energy, retaining the present thermal fit. The candidate is
ready for a separate production decision; this review does not change curated.
A complete NASA-thermal/ATcT alternative is also evaluated. Its additional
thermal effect is smaller than the energy correction and is retained as a
reference-model sensitivity, not a physical uncertainty interval.

The original :doc:`ch2o_source_review` and its execution receipts remain intact.
The source assessment establishes the historical energy discrepancy and
limitations of AYTY as a complete high-temperature thermal replacement. The
present work adds continuous candidates and actual equilibrium solves on the
named ``curated-cos-v1`` background. It does not adopt AYTY or alter any other
molecular potential, elemental inventory, species list or atomic reference.

Energy and caloric construction
-------------------------------

For C(g) + 2 H(g) + O(g) -> CH2O(g), the atom-balanced ATcT 1.228 energy is
1495667 J/mol. This uses Hf(0) = 711.370, 216.034, 246.843 and -105.386 kJ/mol
for C, H, O and formaldehyde. The retained JANAF source anchor, reconstructed
consistently from Hf(298) and H(0)-H(298), is 1502104 J/mol. Thus
delta D0 = -6437 J/mol. The separate printed JANAF Hf(0) column gives 1502102
J/mol; its 2 J/mol rounding difference is preserved in the source record.

The primary candidate changes only the 1/T coefficient of the inherited
five-coefficient formation ln K by delta D0/R. Reaction G and H increase by
6437 J/mol; entropy and heat capacity are unchanged. No fit or extrapolation
to zero kelvin is performed. This is a source-anchor correction: the inherited
finite-temperature fit is not claimed to reproduce every original JANAF caloric
node or provide an exact independently recovered D0. The source review records
a sampled reaction-Cp residual up to 3.545607 J/mol/K.

The second candidate uses the exact 1000--6000 K NASA9 molecular-minus-atoms
coefficients, restricted here to 1500--4000 K. Native NASA energy uses
R = 8.314510 J/mol/K; removing native H(0) and inserting the ATcT D0 uses the
explicit offset D0(ATcT)/R(SI) - D0(NASA)/R(NASA). Dimensionless thermal
functions are retained and calorics use R(SI) = 8.31446261815324 J/mol/K.
Molecular and atomic thermal references both change in this alternative, so
the comparison is not attributed solely to molecular thermal error.

Both functions support scalar, vector, JIT and automatic derivatives with
explicit domain guards. At 2501 temperatures separated by 1 K, independent
source reconstruction and current-factory replay agree below 1e-12 in G/(RT).
Analytic and automatic H/S/Cp derivatives agree to numerical precision.

.. list-table:: Formation constants relative to current curated
   :header-rows: 1

   * - Temperature (K)
     - Retained thermal + ATcT
     - NASA thermal + ATcT
   * - 1500
     - 0.59682654
     - 0.56868726
   * - 3000
     - 0.77254549
     - 0.76334488
   * - 4000
     - 0.82403005
     - 0.79502345

These are formation-constant ratios. The finite abundance responses below
come from solving the coupled equilibrium system.

Finite replacement results
----------------------------

The existing curated-cos baseline supplies 75 converged conditions per network:
1500, 2000, 2500, 3000 and 4000 K; five pressures from 1e-6 to 100 bar; and
reference C/O, 1 and 1.5. Its species, elements, matrix, budgets and complete
potentials are checked before reuse. The two networks contain 537 species with
ions and 421 without. Baselines are explicitly reused; they are not counted as
new solves.

Each candidate is solved independently from cold starts at every condition in
both networks: 300 new endpoints in total. There are also 18 tighter cold
controls, including fresh current baselines, and 10 finite interpolation points
for the primary candidate. Six FastChem comparisons cover current and both
candidates at two ion-inclusive conditions using identical thermodynamics.
Neutral FastChem comparisons are not included.

All endpoints converge. The primary candidate/current CH2O mole-fraction ratio
spans 0.5968265--0.8240359. The NASA-thermal alternative/current ratio spans
0.5686873--0.7950300. At the maximum primary absolute response (2500 K, 100 bar,
C/O = 1.5), CH2O falls from 3.80120e-8 to 2.78892e-8, a change of -1.01229e-8.
CO increases by about 9.57309e-9 at the same condition. At reference C/O,
2500 K and 100 bar, the carbon fraction in CH2O falls from 4.64270e-5 to
3.40631e-5. These fractions are dimensionless, not percentages.

The NASA-thermal control changes CH2O by an additional factor
0.952852--0.989091 relative to the energy-only candidate. Across all species
its maximum additional absolute mole-fraction response is 9.07161e-10.
These are full-grid comparisons, not extrapolations from selected controls.

H/C/O closure is below 2.71e-13 relative, mass-action residual below 1.65e-12
in natural logarithms, and ion charge residual below 1.96e-14. The maximum
relative closure residual over all elements is 1.481e-6 in a trace inventory;
it is not described as H/C/O error. Selected tighter solves change CH2O ln x
by less than 1.50e-13 and any absolute mole fraction by less than 4.26e-14.
FastChem agreement is better than 8.74e-9 dex over all species at the selected
points. Same-model Gibbs comparisons include first-order feasibility corrections.
Passing their numerical tolerance does not resolve every tiny Gibbs difference
or independently validate physical thermochemistry.

Scientific scope and next decision
-----------------------------------

The evaluated energy discrepancy supports the primary correction. The thermal
control confirms that the energy change dominates this response; it does not
establish which complete thermal model is physically exact. Retaining the
thermal fit is a proportionate first update, with its recorded caloric residual
and original source limitations explicit. A later thermal revision should
resolve high-temperature state completeness and calorics, rather than infer
them from partition-function agreement or a line-list temperature label.

The ATcT molecular uncertainty is not a balanced-reaction uncertainty; no
covariance or synthetic confidence band is assigned. The da Silva 2006 abstract
is corroborating context, not an independent ATcT error estimate. AYTY nuclear
spin, isotope and cutoff limitations remain as recorded in the source review.
The hot hydrogen-rich equilibrium grid does not establish low-temperature
photochemical behavior, non-solar atmospheres, transport, condensation or opacity.

Reproduction
------------

Run the archived inputs offline with a CPU JAX installation::

   PYTHONPATH=src JAX_PLATFORMS=cpu python mceg/reviews/ch2o_candidate.py --build
   PYTHONPATH=src JAX_PLATFORMS=cpu python mceg/reviews/ch2o_equilibrium_impact.py \
     --output-dir results/ch2o-candidate-replay --fastchem-executable /path/to/fastchem
   PYTHONPATH=src JAX_PLATFORMS=cpu pytest tests/model_evidence/ch2o_candidate_test.py \
     tests/model_evidence/ch2o_equilibrium_impact_test.py

``tests/data/validation/ch2o_candidate/20261105`` stores the continuous model,
validation and actual-input receipt; ``ch2o_equilibrium_impact/20261105`` stores
the new endpoints, controls, diagnostics and receipt. Numerical reconstruction,
finite substitution, scientific preference and unknown complete physical accuracy
are separate assessments in ``mceg/contracts/ch2o_candidate.yaml``.

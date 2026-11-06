HO2: exact NASA/ATcT candidate and finite substitution
======================================================

The NASA thermal functions with balanced ATcT 1.172 energy are a
scientifically preferable replacement for the inherited HO2 fit within
1500--4000 K. The old JANAF formation energy is superseded by inspected
modern evaluations. The continuous model and finite substitution are now
verified; this study changes no production data or default factory.
A separately authorized curated update can use this exact representation
without a new fit. The evidence does not require obtaining every historical
paper before that bounded decision.

This completes the continuous-candidate and substitution calculation proposed
in the :doc:`dated source review <ho2_source_review>`. That source record
remains a description of the earlier investigation. Full physical accuracy
and a correlated uncertainty band remain unknown.

Construction and source conventions
-----------------------------------

For H(g) + 2 O(g) -> HO2(g), the same ATcT 1.172 evaluation gives
Hf(0) = 216.034, 246.844 and 15.06 kJ/mol for H, O and HO2. Thus
D0 = 694.662 kJ/mol. The molecular uncertainty alone is not a reaction
uncertainty: the network covariance is unavailable. No uncertainty is
inferred from a displayed zero in the H uncertainty column.

The inherited fit follows JANAF Hf(298.15 K) = 2.092 kJ/mol, whereas
NASA uses 12.020 kJ/mol and ATcT 1.172 evaluates 12.15 +/- 0.14 kJ/mol.
Burgess 2016 independently documents the obsolete JANAF evaluation and
selects 12.30 +/- 0.35 kJ/mol. These evaluations are not averaged. Source
inspection and transcribed values are retained in the preceding source
packet, including the access limits on historical ancestry.

The candidate subtracts the exact NASA9 coefficients of H and twice O
from HO2, using the single 1000--6000 K source segment throughout the
bounded 1500--4000 K domain. Header H(298)-H(0) values give the zero-K
anchors; no polynomial is evaluated at zero K. Only the enthalpy integration
constant changes. In particular,

.. math::

   h_{\rm candidate}(T) = h_{\rm NASA}(T)
     - \frac{\Delta(D_0/R)}{T},\qquad
   \Delta(D_0/R) = \frac{694662}{8.31446261815324}
     - \frac{694682.762}{8.314510}
     = -2.02096231694\ {\rm K}.

Here ``h = Gformation/(RT) = -ln K`` at the 1 bar standard state.
NASA's native dimensionless thermal functions are retained; caloric outputs
use modern SI R. Treating the NASA energy with modern R would silently
change the offset by about 0.476 K. A molecular-only Hf replacement would
also mix reference conventions. The candidate balances the complete
reaction and inserts D0 once.

Scalar calls reject temperatures outside the bounded domain. Traced JAX
scalar and array calls return NaN there. A 2501-temperature, 1 K grid checks
independently evaluated source H/S/Cp against the exact reaction and JAX
first/second derivatives. The maximum dimensionless formation error is
6.04e-14, the molecular G error 5.83e-10 J/mol, and the reaction enthalpy
derivative error 8.15e-10 J/mol. These are numerical reconstruction errors,
not physical uncertainty estimates.

Same-energy thermal diagnostic
------------------------------

JANAF is used only at its archived original table knots. Each diagnostic
uses the identical balanced ATcT D0 and NASA atomic thermal functions,
so the residual compares molecular thermal functions. No continuous
JANAF interpolation or external-state completeness bound is introduced.

.. list-table:: Formation-equilibrium constant ratios
   :header-rows: 1

   * - Temperature (K)
     - NASA/ATcT / current
     - JANAF same energy and atoms / NASA/ATcT
   * - 1500
     - 0.446393
     - 1.002276
   * - 2000
     - 0.549343
     - 1.001873
   * - 2500
     - 0.623138
     - 0.999090
   * - 3000
     - 0.678093
     - 0.994150
   * - 3500
     - 0.720649
     - 0.987482
   * - 4000
     - 0.755162
     - 0.979916

The known energy defect dominates the contrast with the current fit.
The remaining thermal difference ranges from +0.23% to -2.01% at these
knots. Related evaluated sources need not be statistically independent;
this contrast is not a confidence interval or a bound on omitted states.

Finite equilibrium response
---------------------------

Both current ``curated-oh-v1`` catalogs are fixed: 537 species including
ions and 421 neutral species. Each has 75 cases at 1500/2000/2500/3000/4000 K,
1e-6/1e-4/1e-2/1/100 bar and C/O = 0.5888436554/1/1.5. All current
potentials, species, element matrices and budgets are checked against the
archived OH-adoption endpoints before reuse. Only the ``H1O2`` potential
changes. All 150 candidate endpoints are newly solved, from cold starts.

The HO2 abundance ratio is 0.446393--0.755162 in both networks: a
24.5--55.4% reduction. Its absolute abundance is extremely small in this
hot hydrogen-rich equilibrium grid. At the largest oxygen-inventory
response (4000 K, 1 bar, reference C/O), the ion-inclusive result is

* HO2 mole fraction: 8.91130e-12 -> 6.72947e-12;
* fraction of all oxygen atoms held in HO2: 3.28852e-8 -> 2.48336e-8;
* oxygen share decrease: 8.05155e-9, or 8.05155e-7 percentage points.

The largest absolute mole-fraction response of any species is H2O at
4000 K and 100 bar: +4.731e-12, only +1.754e-8 relative. Neutral-network
results are nearly identical. Correcting the source remains scientifically
motivated even though its bulk composition effect here is tiny. None of
these results establishes that HO2 is unimportant in cooler atmospheres,
photochemistry or kinetic radical networks.

Numerical checks and limits
---------------------------

The primary residual tolerance is 5e-14, with 12 selected cold reruns at
2e-14 across both models and networks. H/C/O inventory relative closure
is below 9.00e-13; all-element relative closure is below 1.465e-6, reflecting
tiny elemental budgets. The charge residual is below 4.96e-14 and
formation-equation residual below 1.85e-12 in natural-log units. Tightening
changes HO2 ln(x) by at most 1.20e-10 and any mole fraction by at most
3.47e-14. The largest visible-species ln(x) tolerance change is 1.463e-6;
these small numerical differences must not be read as physical responses
of arbitrary trace species.

Five chemical-potential interpolation points at each network's highlighted
case preserve constraints and monotonic HO2 response. Same-model Gibbs
exchange checks subtract the first-order inventory residual; sign deviations
are below 4.13e-18 in dimensionless extensive G, within the 5e-14 check
allowance. Since HO2 is a trace constituent, many Gibbs differences are
below numerical resolution. This check is a consistency test, not a resolved
minimum-energy benefit at every case, and never compares absolute G from
different thermochemical models.

Six JANAF original-node substitutions give HO2 ratios to NASA/ATcT of
0.979916--1.002276; the largest resulting absolute composition change is
1.48e-13. They use the same energy and atomic reference. Four independent
FastChem 4.0.3 calculations at two ion-inclusive cases agree with the
ExoGibbs results within 1.193e-8 dex. Cross-solver agreement verifies the
numerical problem; it cannot resolve shared source errors.

Reproduction and MCEG
---------------------

``tests/data/validation/ho2_candidate/20261105`` contains the exact model,
source hashes and dense validation. ``ho2_equilibrium_impact/20261105``
contains both compact networks, endpoints, tighter controls, interpolation
paths, JANAF controls, diagnostics and the actual execution receipt.
The receipt preserves the command, environment, input/output hashes and
FastChem binary hash. The review scripts reuse existing solver helpers.

.. code-block:: console

   PYTHONPATH=src python mceg/reviews/ho2_candidate.py
   JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     PYTHONPATH=src taskset -c 12-15 python mceg/reviews/ho2_equilibrium_impact.py \
     --output-dir results/ho2_candidate/new-run \
     --fastchem-executable /tmp/mceg-fastchem-v403-bin
   PYTHONPATH=src pytest tests/model_evidence/ho2_candidate_test.py tests/model_evidence/ho2_equilibrium_impact_test.py

``mceg/contracts/ho2_candidate.yaml`` records completed continuous and
finite-response verification and a scoped preference for subsequent
adoption. ``physical_accuracy`` remains unknown. Correlated ATcT energy
uncertainty, high-state completeness and application-specific atmospheric
validation remain follow-ups proportionate to any future accuracy claim,
not additional literature gates on this completed numerical study.

Finite sulfur and empirical FeS saturation
============================================

``examples/metal_silicate/sulfide.py`` adds an example-local finite sulfur
calculation. It leaves the audited dry and GCE source calculations unchanged.
Its runnable model, ``finite_sulfur_empirical_scss_analytic_control_v1``, is
a manufactured numerical control with synthetic SCSS. It establishes local
equation closure and branch handling, not an experimental partition law,
liquid stability, or a calibrated sub-Neptune prediction.

Declared components and exchange
----------------------------------

The finite element order is Mg, Si, Fe, O, H, S. The fixed phases contain:

* gas: H2, H2O, H2S;
* melt: MgO, SiO2, FeO, H2, H2O, FeS, FeSO4;
* metal: Fe, H, S;
* separate sulfide: pure FeS, with its physical state declared in ``SCSS``.

Mg and Si remain finite host budgets. Fe, O and H re-equilibrate as sulfide
appears; each mole of separate FeS consumes one mole each of Fe and S.
The fixed host requires positive Mg/Si/Fe/O/H. Exactly zero S is supported:
all four S-bearing solution components and the separate sulfide disappear
exactly, and only the four nonsulfur reactions remain. This example has no
inert He, SiH4, H/OH, SO2, metal Si/O, competing host solids or carbon/nitrogen;
it therefore does not replace the larger finite-reference or source network.

The caller supplies ``partition_residual(T_K, P_bar, amounts_mol)`` in this
fixed order, with dimensionless residuals for:

#. FeO(melt) + H2(gas) = Fe(metal) + H2O(gas)
#. H2(gas) = H2(melt)
#. H2O(gas) = H2O(melt)
#. H2(gas) = 2 H(metal)
#. H2S(gas) + FeO(melt) = FeS(melt) + H2O(gas)
#. FeS(melt) = Fe(metal) + S(metal)
#. FeS(melt) + 4 H2O(gas) = FeSO4(melt) + 4 H2(gas)

These reactions independently span the solution-phase element null space.
An empirical metal/silicate partition law belongs in this exchange callback;
SCSS alone cannot determine unsaturated gas/melt/metal partition. All amounts
use the documented ``SPECIES`` order. The callback returns seven entries,
with finite values required on the active reactions. At S=0, it must support
exactly zero sulfur components without failing on inactive logarithms.

The control reconstructs ideal solution standards from a known positive
composition at 1873 K and 1 bar. All gas fugacities use the ideal mixing
approximation and a 1 bar reference. Its dimensionless standards have no
calibrated temperature dependence, pressure work, or alloy interactions.
1873 K is a numerical reference, not evidence for the declared liquids.

Saturation convention and phase branches
------------------------------------------

``saturation(T_K, P_bar, amounts_mol)`` is evaluated at each trial and again
at the final composition. It returns ``SCSS(ppm_s, sulfur_state, mass_basis,
calibration_id, phase_state)``. A real calibration must supply a justified
pure-FeS solid/liquid identity, source, composition domain and uncertainty.
The control declares ``phase_state="conditional"``. An Fe-Ni-Cu sulfide liquid
cannot be substituted without its own composition and partition equations.

``melt_sulfur_ppm`` compares the computed concentration directly on the
selected SCSS basis. ``sulfur_state="sulfide"`` counts S in FeS; ``"total"``
counts S in both FeS and FeSO4. The dry mass denominator excludes dissolved
H2 and H2O components, and the hydrous denominator includes them. Both
include the full mass of sulfur-bearing melt components. These are explicit
example conventions: a calibration using a different denominator needs its
own conversion before entering this callback.

For this two-oxidation-state melt, ``scss_total_ppm`` implements
``SCSS_total = SCSS_sulfide / (1 - sulfate_fraction)`` while retaining the
same denominator. It rejects the undefined 100%-sulfate conversion. A
dry/hydrous conversion is a separate mass conversion, not this speciation
correction. At zero S the measured concentration is exactly zero, so no
undefined sulfate fraction is needed by the control.

``solve_branches`` evaluates both explicit branches from every initial
composition:

* Absent FeS is fixed to exactly zero. The solver closes finite budgets and
  unsaturated exchange, then requires computed melt sulfur at or below SCSS.
* Present FeS adds one linear amount and the saturation equation. Negative
  FeS roots remain available as rejected diagnostics. No independent FeS
  chemical-potential equality is imposed alongside empirical SCSS.

Only solution components use logarithmic amounts. A positive initial guess
for a component newly activated after S=0 seeds the solve and does not add
material to the accepted state. Optimizer success is recorded separately
from acceptance. A fresh final callback evaluation requires nonnegative
amounts, exact zero support, maximum relative positive-element residual
below ``1e-9``, maximum reaction residual below ``1e-8``, and the appropriate
saturation equality/inequality within ``1e-8`` in log concentration ratio.
The public ``evaluate`` function independently checks supplied amounts.
It rejects nonpositive or nonfinite temperature and pressure before calling
the partition or saturation model. Active solution amounts are searched in
``-650 <= ln(amount / total elemental amount) <= 10``; initial guesses
outside that numerical range raise an explicit error. Positive trace S
budgets, including the ``1e-50`` mol regression control, remain finite and
are never replaced by a concentration floor or silently removed.

Empirical SCSS does not supply a common Gibbs energy. Accepted roots are
therefore not ranked by Gibbs energy, and no global equilibrium or physical
hysteresis is inferred. Metal absence and host solids are not candidate
branches of this sulfur control.

Reproduction and continuation
-------------------------------

Run the offline reference and its regression tests with::

    python examples/metal_silicate/sulfide.py
    pytest -q tests/unittests/examples/metal_silicate_sulfide_test.py

The script reports every attempted branch, its amounts and independent
residuals. The manufactured saturated state has 0.15 mol of separate FeS;
the same budgets admit a supersaturated absent root, which is rejected.

``continue_sulfur`` takes a sequence of total-S budgets while holding the
other finite budgets fixed. Run it again with the reversed sequence. It
retains all raw starts/roots, carries each distinct accepted composition to
the next point, and keeps the original initial compositions as independent
starts. Continuation order never selects a preferred root. Tests cover
forward/backward phase disappearance, exact-zero S, multiple starts,
amount scaling, independent Fe/S accounting, and equivalent sulfide/total
and dry/hydrous SCSS conversions. An unavailable accepted root remains a
numerical failure, not evidence for physical hysteresis.

Physical integration still requires a selected unsaturated partition
calibration and an SCSS/speciation calibration with compatible host,
redox, concentration and phase conventions. Neither is fabricated by this
control. In particular, broad pressure/temperature coverage does not justify
an FeO-poor, strongly reduced host outside the chosen calibration.

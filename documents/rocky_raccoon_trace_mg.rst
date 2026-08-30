Rocky Raccoon-like positive trace magnesium
============================================

Purpose
-------

This example exercises the ExoGibbs boundary exposed by a deep-layer
Rocky Raccoon-like calculation in ExoExamples.  It is a one-layer provider
regression, not a reproduction of the full structure model or of the
published radii in Misener et al. (2026).

The state has

.. code-block:: text

   T  = 1433.764595 K
   P  = 8796.093022 bar
   Mg = 2.415508476e-12

in a normalized ``H, Mg, Si, O, C, charge`` inventory.  Positive magnesium is
close to the capacity of the active condensates, so the final zero-barrier
polish must resolve rather than discard it.

Chemical network
----------------

The example selects the 70 gas species and 14 canonical condensates used by
the Rocky Raccoon-like ExoExamples model from the packaged ion-inclusive
FastChem4-format thermochemical tables.  ExoGibbs minimizes this explicit
network directly.  Neutral atomic reference gases and a free-electron gas
species are not added.  The ``e-`` formula-matrix row is instead a charge
constraint with zero inventory; positively and negatively charged gas
species jointly satisfy it.

As in the paper-extrapolated ExoExamples mode, condensate coefficients remain
eligible outside their tabulated upper validity bounds.  This is an explicit
policy of this comparison state, not a general recommendation for atmospheric
calculations.

Acceptance checks
-----------------

The current public ``solve_profile`` route converges with positive magnesium
in both gas and condensates.  The example checks

* finite, non-negative gas and condensate amounts;
* reconstruction of every nonzero elemental inventory to relative
  tolerance ``1e-8``;
* charge residual below ``1e-14``;
* normalized gas mole fractions; and
* strictly positive gas-phase and condensed magnesium.

The current solution selects ``SiO2(s,l)`` and ``MgSiO3(s,l)``.  That exact
support is reported for diagnosis but is not an acceptance condition: a
future solver may find another physically equivalent active set.

Provider unit regressions reuse this exact network at four neighboring
deep-atmosphere boundaries. They require the rainout transport scale to remain
one in the ordinary caller gauge and certify all final KKT blocks at ``1e-8``.
Together they cover an initializer-only finite-barrier rescue, a fully
certified root found when the exact optimizer reaches its evaluation limit,
and a rank-deficient four-phase initializer whose physical solution lies on a
different two-phase basis. The fourth state has
:math:`T=1173.1942732095774` K, :math:`P=4132.5213914599017` bar, and a
normalized Mg inventory of :math:`7.5890\times10^{-17}`. Its finite-barrier
solve fails at the trace-capacity boundary even after the initial support is
reduced to a full-rank basis.

The rank-deficient case is handled without a species-specific rule. ExoGibbs
constructs a bounded, deterministic portfolio of nonnegative full-rank bases
that preserve the initial condensed inventory, visits them in canonical
catalog order, and selects a basis only through the existing exact KKT audit.
A local root returns immediately to the outer inactive-phase closure; final
acceptance still requires its complete audit. The search visits at most 32
bases and shares the active-set function-evaluation budget. These extra states
are not added to the documented one-layer performance workload.

For the trace-capacity state, the lifecycle preserves the finite-barrier input
before PDIPM changes it. If the established finite-barrier terminal-state route
is unavailable, one bounded zero-barrier closure may use that preserved state
as an initializer. The failed restoration state is retained for diagnostics
but is not used as the exact-solver initializer. This fallback is selected by
generic restoration-status and capacity-scale gates, not by a species name,
temperature, or pressure.

Diagnostics identify the initializer source as
``pre_pdipm_finite_support_state`` and record the attempt separately in
``pre_pdipm_zero_barrier_fallback``. It does not relax the zero-barrier
equations, the catalog-wide inactive-phase closure, the caller-gauge audit, or
any final ``1e-8`` KKT tolerance.

Run the example
---------------

Download :download:`demo_rocky_raccoon_trace_mg.py
<../examples/comparisons/demo_rocky_raccoon_trace_mg.py>` or run this one
liner from the repository root:

.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/comparisons/demo_rocky_raccoon_trace_mg.py

The script prints a JSON physical audit.  No ExoExamples checkout, external
FastChem executable, network access, plot, or output file is required.

Responsibility boundary
-----------------------

ExoGibbs owns the gas--condensate equilibrium, zero-barrier acceptance, and
rainout inventory returned for this layer.  ExoExamples owns the coupled
pressure--temperature integration and the larger Rocky Raccoon-like forward
model.  ExoEOS and ExoJAX are outside this single-layer regression.

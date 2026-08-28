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

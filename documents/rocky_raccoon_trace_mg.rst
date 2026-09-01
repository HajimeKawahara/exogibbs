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

Provider unit regressions reuse this exact network for a set of cold one-layer
deep-atmosphere inputs. They require the rainout transport scale to remain one
in the ordinary caller gauge and certify all final KKT blocks at ``1e-8``. The
accepted-parent transition regressions start from an accepted parent gas state
and record the source temperature, pressure, and incoming inventory as
numerical provenance. They propagate only gas amounts and total gas, matching
the rainout contract; they do not propagate condensate amounts or support.
Together they cover an initializer-only finite-barrier rescue, a fully
certified root found when the exact optimizer reaches its evaluation limit,
and a rank-deficient four-phase initializer whose physical solution lies on a
different two-phase basis. One trace-capacity state has
:math:`T=1173.1942732095774` K, :math:`P=4132.5213914599017` bar, and a
normalized Mg inventory of :math:`7.5890\times10^{-17}`. Its finite-barrier
solve fails at the trace-capacity boundary even after the initial support is
reduced to a full-rank basis.

Adjacent states at pressure steps 378 and 380 expose backend-sensitive finite
globalization classifications. For a fixed input, backend roundoff can lead to
``NORMAL_DUAL_STEP_FAILED``, ``RESTORATION_LINE_SEARCH_FAILED``, or
``RESTORATION_LOCALLY_INFEASIBLE``. None of these terminal states is accepted;
each may only initialize the capacity-gated exact closure from its preserved
input.

At pressure step 397 and the adjacent later boundary, PDIPM uses one full-rank
representation selected from the initial rank-deficient support. Before exact
polishing, the lifecycle expands the selected terminal or pre-PDIPM
initializer back to that support envelope without changing any amount. The
added candidate phases therefore start at exactly zero amount and the current
condensate burden is preserved. If the first basis does not reach a local root,
the bounded basic-support portfolio tries the bounded set of untried feasible
bases; no phase name, temperature, or pressure appears in the solver policy.

The lower-temperature boundary at :math:`T=1029.4199562443821` K and
:math:`P=2709.5257545305749` bar has normalized Mg inventory
:math:`2.0670115092041315\times10^{-19}`. There, every tested full-rank basis
that preserves the initial condensed inventory misses a local exact root.
The support-release initializer portfolio therefore explores proper faces of
the selected basis. It keeps retained amounts, zeros removed amounts, and lets
the existing exact gas--condensate solve determine a new partition subject to
the unchanged total inventory equation. A released local root then returns to
the ordinary inactive-phase closure; it is not accepted on its own. The
bounded search uses only formula geometry, conservative capacity ordering,
catalog order, and the shared work budget.

The accepted-parent warm path reaches a still lower-temperature boundary
at :math:`T=868.75060835990814` K and
:math:`P=1438.5022418599176` bar, with normalized Mg inventory
:math:`6.424866149810836\times10^{-24}`. Its finite-barrier endpoint selects a
full-rank basis from a rank-deficient support, but trace-scale element
potentials place the unregularized exact solve outside a useful local basin.
The normalized-linear exact formulation therefore tries the existing
capacity-regularized gas--potential initializer. The exactly zero charge target
and signed charge row remain in that solve because its budget equations are
linear. The log-domain fallback likewise retains charge as a scaled linear row
while using logarithmic residuals for positive monotone element budgets. The
regularized local root drops a
phase whose inactive driving immediately reopens it. The generic initializer
selector therefore defers that root and uses the protected unregularized
attempt, which reaches the accepted state. Both routes remain subject to the
unchanged full-catalog and caller-gauge physical audits.

At :math:`T=774.98996736332037` K and
:math:`P=1001.7919754251911` bar, the ordinary capacity-regularized solve keeps
the preserved selected-support condensate seed and reaches a local KKT root
with initializer-relative scaling. No unit-scaled restart or condensate reset
is needed. Ordinary inactive-phase closure then adds phase 8, the existing
rank-one pivot selects the independent support ``(1, 8)``, and the complete
physical and caller-gauge audits pass.

The colder accepted-parent regression at :math:`T=480.4777949967222` K and
:math:`P=179.6370128930636` bar has normalized Mg inventory
:math:`2.40376970\times10^{-46}`. There, the initializer-relative,
capacity-regularized solve reaches a finite status-0 evaluation-limit state
with positive active amounts but without a local KKT certificate. One guarded
dimensionless-unit restart uses that terminal gas, condensate, total-gas,
potential, and support state without a condensate reset or alternative
inventory partition. The restarted root enters ordinary phase
closure, which adds phase 8 and accepts support ``(1, 8)`` only after the
unchanged full physical and caller-gauge audits pass.

At :math:`T=475.01010900904657` K and
:math:`P=172.55859783339542` bar, the normalized Mg inventory is
:math:`6.300502379398082\times10^{-47}`. The selected basic support does not
reach a local root, so the deterministic alternative-basic-support portfolio
first exhausts its normalized-linear candidates. Its mixed log/linear pass
then selects support ``(1, 8)``: positive H, Mg, Si, O, and C budgets remain
in log space, while the exactly zero signed charge budget remains linear. The
mixed pass uses the existing capacity-regularized gas--potential initializer,
shares the same bounded portfolio budget, and is accepted only by the
unchanged full physical and caller-gauge audits.

At :math:`T=386.57556568831939` K and
:math:`P=83.689430815806617` bar, the normalized Mg inventory is
:math:`2.139116395339677\times10^{-56}`. The finite-barrier endpoint retains a
trace-incompatible condensate burden: the basic-support LP fails, and the
bounded burden-preserving alternative portfolio does not reach a local root.
The first canonical feasible alternative therefore supplies only a
builder-produced support-release initializer. Within two protected solver-call
allowances, a mixed positive-log/signed-linear solve releases support
``(1, 4)`` to the proper face ``(1,)``. Ordinary inactive-phase closure then
adds phase 8 and accepts final support ``(1, 8)`` only after the unchanged full
physical and caller-gauge audits pass. This recovery is selected by support
geometry, portfolio outcome, and available work, not by a species name,
temperature, or pressure.

The deeper trace-depletion boundary at
:math:`T=203.06986826073876` K and :math:`P=8.7214641233652035` bar
has normalized Mg and Si inventories of
:math:`1.3681948091591687\times10^{-93}` and
:math:`4.137051394836369\times10^{-84}`. The basic-support LP selects
``(0, 1, 5)`` while preserving the finite-barrier condensate burden, but no
tested burden-preserving basis reaches a local root. A normalized-linear solve
of alternative basis ``(0, 1, 8)`` terminates with a negative amount only for
phase 0. The solver uses that sign pattern solely to select the original
nonnegative builder-produced basis as a support-release source; it never
adopts the rejected terminal amounts. The mixed positive-log/signed-linear
proper-face portfolio then accepts ``(1, 8)`` under the unchanged physical
audit. If the indicated proper face had already been tried, the generic source
selector would retain the selected LP basis instead. The alternative search,
support release, and outer closure share one hard work budget with one complete
call allowance protected for each downstream stage.

The next default-column boundary occurs at
:math:`T=157.89357053396711` K and :math:`P=3.7871329378560565` bar, with
normalized Mg and Si inventories of
:math:`6.831754190721877\times10^{-112}` and
:math:`8.328995133274878\times10^{-119}`. A direct gas-only warm solve cannot
enter the basin of the physical support ``(9,)``. Uniform continuation in
temperature, pressure, and inventory is not used: empirical success is not
monotone in that step size.

Instead, the rainout scheduler uses the accepted source problem only to form
one inventory midpoint at the exact target temperature and pressure. Positive
rows are interpolated logarithmically, while zero endpoints are interpolated
linearly. The midpoint converges on support ``(1, 8)`` and must pass the same
lifecycle and floorless budget audits as an ordinary layer. Only its gas state
then initializes a second solve at the exact target inventory. That solve
selects support ``(9,)`` and passes the unchanged full physical and
caller-gauge audits. The midpoint condensates, support, and rainout output are
discarded; only the exact target is propagated. The route adds at most two
lifecycle calls, contains no species-specific condition, and falls back cold
if the midpoint or exact target retry is rejected.

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
as an initializer. The failed finite-barrier state is retained for diagnostics
but is not used as the exact-solver initializer. This fallback is selected by
generic eligible terminal-status and capacity-scale gates, not by a species
name, temperature, or pressure.

Both the trace-capacity gate and the gas-capacity regularizer derive their
monotone row mask from the joint gas and condensate formula catalogs. The
signed charge row is retained in conservation but cannot define a capacity
ceiling because positive and negative carriers can offset one another.

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

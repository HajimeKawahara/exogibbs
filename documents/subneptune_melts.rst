Finite external MELTS--metal--gas controls
============================================

``examples/metal_silicate/run_melts_reference.py`` connects the optional
supplied-composition MELTS evaluator to a finite local ExoGibbs solve. Its
model identifier is ``melts_ma_fe_si_o_h_finite_conditional_v1``. This is a
conditional mechanism calculation: a converged local chemical root does not
establish experimental calibration or a stable assemblage.

The audited ``completed_dry``, ``source_full``, and ``gas_exchange`` controls
are unchanged. ``ChemicalSetup`` and the existing application API are also
unchanged; the host numerical solve and full-phase callbacks are example-local.

Basis and common potentials
-----------------------------

The host retains every positive endmember in the first saved ExoEOS MELTS
liquid composition. Mg, Al, Ca, Na, K, Ti, Cr and P budgets are preserved,
together with Si, Fe, O and H. That fixture provides an initial composition,
not an interpolated property model: every chemical trial calls the actual
supplied-composition evaluator. Zero C and unsupported components remain
exactly zero. Finite H2, He and Fe--Si--O--H alloy amounts are added to define
the complete local inventory. He is an assumed inert gas.

The host basis is the MELTS endmember basis, including Mg2SiO4, Fe2SiO4 and
Fe2O3, not the source association basis or a relabeled oxide activity vector.
The record exports the formula matrix, component and element orders, molar
amount convention, model and backend identities, actual import paths, file
hashes, and known domain restrictions. Host arrays passed to ExoEOS retain
its full 19-component order even when some endmembers are absent.

The full MELTS chemical potentials already contain ideal and excess mixing
and condensed pressure work. They are converted to the common
:math:`R=8.31446261815324` J/(mol K). ExoGibbs adds only the reduced-H2
construction from :doc:`subneptune_hydrogen`: its scalar host/H2 dilution
changes every host potential reciprocally. The existing MELTS H2O potential
is used without imposing a second H2O-solubility equation.

The alloy is ExoEOS ``MaFeSiOHLiquid``. Its native Fe/Si/O/H standard shifts
and activities are combined with the source's absolute standards, including
the inherited Okuchi H standard. H2 dissolution uses the named older
Hirschmann/Seo sensitivity law. The gas network contains the supported source
species plus He, SiH4, H and OH. The additional gas NASA7 coefficients and
independent 2350 K values are frozen in ``reduced_gas_reference.json`` from
the installed ``thermochem==0.9.0`` Burcat/Ruscic dataset, identified by its
SHA256. Formation-enthalpy/absolute-entropy standards are used, not FastChem's
atomic-reference potentials. The source and additional-gas values are
explicitly converted from their recorded R to the common R.

This nominal common convention does not prove experimental exchange-standard
alignment across these sources. No phase-specific offset is fitted to force
one chosen composition. Independent exchange data, revised host-specific H2
calibration, gas-species convergence, and alloy pressure response remain
acceptance requirements for a physical prediction.

Local solve and phase branches
--------------------------------

``full_potential.py`` reuses ``local.build_problem`` for static phase and
exact-zero element support. A phase callback receives T in K, P in bar and
its current component amounts in mol, and returns complete ``mu_rt`` and
extensive ``gibbs_rt``. ``melts_coupled.py`` converts pressure to Pa for the
external provider. It rejects a changed composition, imposed oxygen buffer,
unavailable present-component potential, or a mismatched model, basis, T/P
or gas constant.

The host SciPy solver evaluates the complete residual and numerical Jacobian.
It adds no further mixing or gas pressure term to a full callback. Every
phase must satisfy the Euler identity, and final amounts are independently
re-evaluated with a fresh backend call before acceptance. Positive elemental
budgets require relative residual below 1e-9 and dimensionless independent
reaction residual below 1e-8. Results include absolute component/phase amounts,
phase elemental contributions and elemental potentials. These are local
parcel amounts, not a planetary atmospheric column or pressure closure.

``--metal-absent`` removes metal components exactly while retaining the same
total element budgets. Both present and absent branches can have local roots;
one must additionally check phase insertion and stability. Helpers provide
pure-phase insertion energy and the exact minimum over an unconstrained
ideal-solution simplex. The latter is not applicable to the nonideal native
alloy. Components containing zero-budget elements must be removed before
an insertion calculation. The example does not assert an absent alloy's
stability by testing one guessed composition.

The final host competing-solid assemblage and nonideal absent-alloy insertion
minimum have not been validated. The output explicitly marks phase acceptance
as unchecked. A below-liquidus root can only be a phase-suppressed control.
No global Gibbs minimum, liquid stability, or derivative through a phase
change is promised.

Reproduce with the separately installed runtime
-------------------------------------------------

Use ExoEOS commit ``0c85dfe`` or a compatible supplied-composition evaluator,
and its hash-pinned alphaMELTS 2.3.2 / rhyolite-MELTS 1.0.2 runtime. The
worker Python environment needs the dependencies documented by ExoEOS.
There is no download, installation or external dependency on ordinary imports.

.. code-block:: console

   PYTHONPATH=src:/path/to/exoeos/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
     python examples/metal_silicate/run_melts_reference.py \
       --exoeos-checkout /path/to/exoeos \
       --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
       --python /path/to/melts-python \
       --output results/subneptune_taxonomy/melts_present.json

Repeat with ``--metal-absent`` and a separate output file. The default
2350 K / 1 bar conditions are formal extrapolations of the source-host and
H2 assumptions, not a jointly calibrated liquid domain. Changing T does not
reselect the source's frozen 2350 K Shomate branches.

The real-backend present reference converged with maximum elemental residual
6.7e-16 and reaction residual 2.9e-14. Phase amounts were approximately
0.812636 mol host including dissolved H2, 0.0975854 mol alloy, and
0.0350556 mol gas. The same-budget metal-absent root had approximately
0.770049 mol host and 0.128315 mol gas, with every metal component exactly
zero; elemental and reaction residuals were below 4.5e-16 and 1.8e-14.
Fresh evaluation of these archived amounts gives reaction residuals below
1.4e-14 (present) and 1.8e-14 (absent). Inserting the present root's alloy
composition into the absent root gives :math:`D/(RT)=-2.89230` per mole
of trial alloy; the same-budget difference is
:math:`(G_{present}-G_{absent})/(RT)=-0.225361`. This negative insertion
trial disproves alloy absence within the formal model. The metal-absent
root is an unstable phase-suppressed diagnostic, not a second stable state.
The trial does not establish the present root's stability against every
competing composition or phase, or experimental exchange calibration.

Offline tests independently cover finite H/He/Mg/Si/Fe/O closure, amount
rescaling, exact phase removal, ideal-solution insertion minimization,
rejection of failed roots, fresh final evaluation, common-R conversion,
background-element conservation, and malformed external responses. They
do not substitute for the optional real-backend command or its remaining
physical acceptance gates.

Revalidate archived amounts
-----------------------------

The dated verifier recomputes hydrogen, source S/N and Carbon, and analytic
SCSS chemistry from the archived component amounts. It also recomputes
rejected SCSS branch admissibility and requires agreement with the saved
acceptance flags. Hashes and saved residual fields alone are insufficient:
an element-conserving reaction displacement must fail chemical acceptance.
The original numerical JSON files remain unchanged.

.. code-block:: console

   PYTHONPATH=src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
     python results/subneptune_taxonomy/20260911/verify.py

This default explicitly reports MELTS chemistry as ``not_run``. To replace
its balance-only checks with fresh supplied-composition provider potentials,
including the insertion trial and Gibbs comparison above, run:

.. code-block:: console

   PYTHONPATH=src:/path/to/exoeos/src JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
     python results/subneptune_taxonomy/20260911/verify.py \
       --exoeos-checkout /path/to/exoeos \
       --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
       --worker-python /path/to/melts-python \
       --output /tmp/revalidated_archive.json

CI covers every PR base, including stacked feature branches. A separate job
pins ExoEOS commit ``0c85dfe28353bf70d7d687e49689e74db556c4b0``, asserts the
actual import paths and native hydrogen model, and rejects any skipped
provider integration test. It does not install the external MELTS runtime;
the explicit replay above supplies that separate evidence.

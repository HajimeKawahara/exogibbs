HCP retained-thermal energy adoption
====================================

``curated_hcp.chemsetup()`` adopts the reviewed energy-only correction on
``curated-tio2-v1``. Reaction G and H increase by 62308 J/mol, while S/Cp,
the literal thermal fit, atomic reference and source-fit residual are retained.
Only HCP changes at this stage; species, inventories, charges and all other
chemical potentials are identical to the preceding named preset. Earlier
presets, source-review packets and candidate-execution receipts remain available.
This named adoption does not itself switch the public ``curated`` alias.

Scientific selection
--------------------

The selected CCCBDB formation enthalpy is 213.500 kJ/mol at 0 K versus
JANAF 151.192 kJ/mol, on the retained white-phosphorus elemental reference.
This is the inspected CP-based adjustment to JANAF, not a new independent
HCP measurement. The 2001 calculation (225 kJ/mol with reported +/-8 kJ/mol)
and CCCBDB's reported +/-20 kJ/mol remain separate source sensitivities;
the alternative is not added to this correction. The adopted common-reference
D0=1029.274 kJ/mol is distinct from the 2001 own-atom TAE0=1018 kJ/mol.
The retained physical JANAF D0 is 1091.582 kJ/mol, whereas the formal fit
intercept is 1095.834364 kJ/mol. Their residual remains intact.
The 2024 PES/DMS supplement does not supply complete high-temperature states
or partition functions and is not used to construct new thermals.

No second zero-point, isotope, spin or fitted reference correction is added.
The empirical fit is evaluated only over 1500-4000 K, never extrapolated to
zero temperature. A modern central energy is a scoped source selection;
reported marginal source uncertainties do not give a full correlated
reaction uncertainty or high-temperature physical validation.

Actual production verification
------------------------------

The actual packaged factory receives 150 fresh cold solves: 75 with 537
ion-inclusive species and 75 with 421 neutral species. The grid comprises
1500, 2000, 2500, 3000 and 4000 K; 1e-6, 1e-4, 0.01, 1 and 100 bar; and
C/O=0.5888436554, 1 and 1.5. Baseline hashes, named-factory potentials,
species/formula matrices, elemental budgets and conditions are verified
before the previous 150 endpoints are reused for comparison. They are not
counted as fresh production solves.

A separate construction inserts the reviewed candidate into the new base.
It agrees with the packaged factory over 2501 temperatures within
2.13e-14 in h. Six fresh tighter cold reference solves agree within
3.34e-13 in target ln(x); the all-species maximum is 1.47e-05.
Two independent standalone FastChem comparisons on selected ion-inclusive
conditions agree within 1.78e-08 dex. All solves converge. Maximum relative
HCNOPS closure is 1.26e-10, the all-element trace-relative value is
1.47e-06, charge residual is 1.29e-14, and formation-equation
residual is 1.42e-12 in ln units.

.. list-table:: Incremental solved response and largest elemental-allocation change
   :header-rows: 1

   * - Network
     - HCP mole-fraction ratio minimum / maximum
     - P allocation before / after at the highlighted case
     - Highlighted condition
   * - Ion-inclusive
     - 0.00676526907 / 0.871073277
     - 94.2882432% / 8.16306442%
     - 1500 K, 1 bar, C/O=1.5
   * - Neutral
     - 0.00676526907 / 0.87107328
     - 94.2882433% / 8.16306442%
     - 1500 K, 1 bar, C/O=1.5

These are coupled solved abundance changes relative to ``curated-tio2-v1``, not
analytic equilibrium-constant sensitivities. The summary separately records
cumulative HCP/C2H2/CH2 changes relative to the original ``curated-tio2-v1``
background. A cumulative response must not be mislabeled as the isolated
molecule-only candidate response when earlier stages changed the background.
Neither cross-model absolute Gibbs values nor independent numerical solver
agreement establish which physical source is correct.

Evidence and physical limits
----------------------------

The packaged JSON, its SHA-pinned loader and versioned base enforce the adopted
energy and scope without an MCEG runtime dependency. Scalar, batched, JIT, vmap,
neutral projection, invalid-temperature masking and unchanged S/Cp have focused
unit checks. Offline replay verifies execution hashes, conserved constraints,
strict controls and quoted responses. The shared runner is a small reuse of the
existing production cold solver and independent FastChem comparison routines.

``curated_hcp_adoption.yaml`` has supported scientific-selection and
production-adoption assessments. Complete physical accuracy remains unknown.
1500-4000 K, one-bar standard pressure. CCCBDB is a CP-based JANAF adjustment, not an independent HCP measurement. Its reported +/-20 kJ/mol is not a complete reaction covariance. Retained thermals and high-temperature physical accuracy remain limited.
The ideal-gas network excludes condensation, transport and photochemical
kinetics; numerical closure does not establish those application accuracies.
These limits do not prevent the explicitly scoped energy adoption.

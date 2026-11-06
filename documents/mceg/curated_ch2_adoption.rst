CH2 retained-thermal energy adoption
====================================

``curated_ch2.chemsetup()`` adopts the reviewed energy-only correction on
``curated-c2h2-v1``. Reaction G and H increase by 4948 J/mol, while S/Cp,
the literal thermal fit, atomic reference and source-fit residual are retained.
Only CH2 changes at this stage; species, inventories, charges and all other
chemical potentials are identical to the preceding named preset. Earlier
presets, source-review packets and candidate-execution receipts remain available.
This named adoption does not itself switch the public ``curated`` alias.

Scientific selection
--------------------

The same-version ATcT 1.228 cycle is
711.370+2(216.034)-391.046=752.392 kJ/mol for regular CH2. The retained physical
JANAF D0=757.340 kJ/mol therefore gives G/H+4948 J/mol. The formal fit intercept
R*a=758.027186 kJ/mol is a separate empirical quantity whose residual is retained.
The independent three-state RRHO model is not adopted. Its triplet and singlet
component tables reproduce, but both direct printed-component combination and
the independent reconstruction fail to reproduce the published equilibrated
table. At 3000 K direct combination differs by -2974.938 J/mol in H and
-2.716719 J/mol/K in S. The cause remains unresolved. RRHO extension beyond
3000 K and numerical state-sum convergence do not validate large-amplitude or
Renner-Teller thermals. The 2024 full manuscript/SI demonstrates Q convergence
at 296 K only, without a complete high-temperature state archive. A partial
triplet-only species is not interchangeable with the adopted regular CH2 energy.
NASA-proxy residuals in the earlier thermal sensitivity are not interpreted
as a literal JANAF-table replacement.

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
3.55e-15 in h. Six fresh tighter cold reference solves agree within
7.82e-14 in target ln(x); the all-species maximum is 1.4e-05.
Two independent standalone FastChem comparisons on selected ion-inclusive
conditions agree within 1.78e-08 dex. All solves converge. Maximum relative
HCNOPS closure is 1.26e-10, the all-element trace-relative value is
1.4e-06, charge residual is 3.65e-14, and formation-equation
residual is 1.02e-12 in ln units.

.. list-table:: Incremental solved response and largest elemental-allocation change
   :header-rows: 1

   * - Network
     - CH2 mole-fraction ratio minimum / maximum
     - C allocation before / after at the highlighted case
     - Highlighted condition
   * - Ion-inclusive
     - 0.672509926 / 0.868412416
     - 2.63181867% / 2.28456814%
     - 4000 K, 100 bar, C/O=1.5
   * - Neutral
     - 0.672509926 / 0.868412693
     - 2.63186092% / 2.28460481%
     - 4000 K, 100 bar, C/O=1.5

These are coupled solved abundance changes relative to ``curated-c2h2-v1``, not
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

``curated_ch2_adoption.yaml`` has supported scientific-selection and
production-adoption assessments. Complete physical accuracy remains unknown.
1500-4000 K, one-bar standard pressure. Regular equilibrated CH2 is distinct from partial triplet-only species. ATcT covariance, large-amplitude/anharmonic/Renner-Teller thermals, isotope mixtures and planetary physical accuracy remain unquantified.
The ideal-gas network excludes condensation, transport and photochemical
kinetics; numerical closure does not establish those application accuracies.
These limits do not prevent the explicitly scoped energy adoption.

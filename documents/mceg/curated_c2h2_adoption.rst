C2H2 retained-thermal energy adoption
=====================================

``curated_c2h2.chemsetup()`` adopts the reviewed energy-only correction on
``curated-hcp-v1``. Reaction G and H increase by 1135 J/mol, while S/Cp,
the literal thermal fit, atomic reference and source-fit residual are retained.
Only C2H2 changes at this stage; species, inventories, charges and all other
chemical potentials are identical to the preceding named preset. Earlier
presets, source-review packets and candidate-execution receipts remain available.
This named adoption does not itself switch the public ``curated`` alias.

Scientific selection
--------------------

The same-version ATcT 1.228 neutral-atom cycle is
2(711.370)+2(216.034)-228.790=1626.018 kJ/mol. Compared with physical JANAF
D0=1627.153 kJ/mol, the energy difference raises G/H by 1135 J/mol.
The formal fit intercept R*a=1633.462873 kJ/mol is not a physical atomization
measurement; its source-fit residual is retained. The NASA9+ATcT alternative
changes thermal shape, atomic thermal reference and residual, so its
successful numerical solution does not justify adopting it. The finite aCeTY
partition diagnostics likewise do not establish a complete hot state sum.
Only the reviewed retained-thermal energy alternative is adopted.

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
2.84e-14 in h. Six fresh tighter cold reference solves agree within
4.76e-13 in target ln(x); the all-species maximum is 1.38e-05.
Two independent standalone FastChem comparisons on selected ion-inclusive
conditions agree within 1.39e-07 dex. All solves converge. Maximum relative
HCNOPS closure is 1.26e-10, the all-element trace-relative value is
1.38e-06, charge residual is 4.3e-14, and formation-equation
residual is 1.36e-12 in ln units.

.. list-table:: Incremental solved response and largest elemental-allocation change
   :header-rows: 1

   * - Network
     - C2H2 mole-fraction ratio minimum / maximum
     - C allocation before / after at the highlighted case
     - Highlighted condition
   * - Ion-inclusive
     - 0.913012158 / 0.995672816
     - 11.5809295% / 11.0528711%
     - 2000 K, 1e-06 bar, C/O=1.5
   * - Neutral
     - 0.913012158 / 0.995672822
     - 11.5809449% / 11.0528862%
     - 2000 K, 1e-06 bar, C/O=1.5

These are coupled solved abundance changes relative to ``curated-hcp-v1``, not
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

``curated_c2h2_adoption.yaml`` has supported scientific-selection and
production-adoption assessments. Complete physical accuracy remains unknown.
1500-4000 K, one-bar standard pressure. Marginal ATcT uncertainties do not specify correlated atomization uncertainty. Complete hot rovibrational state-sum convergence, isotope mixtures and atmospheric accuracy remain unquantified.
The ideal-gas network excludes condensation, transport and photochemical
kinetics; numerical closure does not establish those application accuracies.
These limits do not prevent the explicitly scoped energy adoption.

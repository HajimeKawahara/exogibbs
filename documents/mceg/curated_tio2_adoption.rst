Curated TiO2 measured-thermal adoption
======================================

Decision and scope
------------------

Adopt the reviewed ground-state TiO2 RRHO thermal correction on the cumulative
CN/PS base as ``curated-tio2-v1``. The named ``curated-ps-v1`` and earlier presets
retain their definitions. This changes only TiO2 relative to the immediate
PS base. The adoption implements the scientific direction evaluated in
:doc:`tio2_candidate`, with user authorization for the bounded production update.

Within the same rigid-rotor/harmonic-oscillator approximation, measured rotation
and bending are preferable central inputs to the 1973 estimated geometry and
bend. The earlier candidate changes formation K by approximately 9--11% over
1500--4000 K. Selected bend and electronic sensitivities are smaller but do not
bound every omitted effect. The thermal adoption does not establish complete
high-temperature physical accuracy and does not adopt a new energy evaluation.

Packaged representation
------------------------

The package contains the exact reviewed rotational constants, gas fundamentals,
NASA molecular thermal coefficients and reference conventions. Runtime needs no
MCEG installation, downloaded partition table or source-review archive. It adds

.. math::

   \delta G(T) = [G_{\mathrm{RRHO}}(T)-H_{0,\mathrm{RRHO}}]
                -[G_{\mathrm{NASA}}(T)-H_{0,\mathrm{NASA}}]

to the current TiO2 formation potential. The compact analytic expression for
RRHO ``G/(RT)`` is minus the translational and rotational log partition functions
plus the harmonic-oscillator log factors. Its temperature derivatives give the
same H/S/Cp as the independently evaluated candidate.

The source D0 remains 1267.776 kJ/mol. All inherited fit residuals relative to the
NASA thermal proxy, atomic potentials, species, formula matrix and inventories
are retained. No formal zero-temperature limit of the inherited finite-range
fit is assumed. The source-native NASA R convention has already been handled
in the pinned molecular coefficients. No additional zero-point energy, H298
reference shift or excited-state correction is added.

The RRHO inputs use measured ``48Ti16O2`` rotation, gas fundamentals
959/322/944 cm-1 treated as harmonic frequencies, symmetry number two, spin
weight one and JANAF mean mass 79.8788 u. This deliberately retains the earlier
mass convention; it is not an isotope-mixture calculation. The candidate's
H298-fixed control and selected electronic-state sensitivities remain evidence,
not production modifications. See the original candidate for primary-source
access and the distinction between gas fundamentals and matrix data.

Scalar and array thermodynamics, JIT and differentiation are supported on
1500--4000 K at the one-bar standard state. Invalid temperatures return NaN.
Use float64 for the reviewed numerical precision. Package hashes and the pinned
PS-base hash reject unexpected input drift.

Validation and cumulative background
--------------------------------------

The adoption validation compares the packaged potential against the independent
candidate applied to the same cumulative PS base at 2501 temperatures. Separate
checks verify only TiO2 changes, scalar/array and neutral/ion projections, domain
guards, input hashes and analytic versus automatic thermodynamic derivatives.

The execution packet records fresh production equilibria for both the 537-species
ion-inclusive and 421-species neutral gas networks. Each grid uses five
temperatures, five pressures and three C/O values. The immediate comparison is
against the preceding CN/PS production baseline, with all species, elemental
budgets and baseline potentials checked before using its archive. Tighter
candidate controls and standalone FastChem comparisons assess numerical
reproduction separately from physical model preference.

The original TiO2-only candidate archive predates CN and PS adoption and contains
only ion-inclusive endpoints. Its TiO2 potential is unchanged by those background
updates, but its equilibrium abundances need not equal the cumulative production
result. The summary separates incremental TiO2 changes from cumulative changes
relative to ``curated-cos-v1`` and explicitly labels the unavailable neutral
single-candidate abundance comparison. No new numerical execution is attributed
to the original candidate receipt.

Completed production results
----------------------------

All 150 fresh endpoints, six tighter candidate controls and two standalone
FastChem comparisons completed successfully. The maximum 2501-temperature
potential difference from the independently assembled candidate is 2.31e-14
in G/(RT) in both network modes. Target tight-control variation is at most
1.71e-13 in ln mole fraction, and FastChem parity is within 6.75e-9 dex.

.. list-table:: TiO2 response relative to the immediate PS base
   :header-rows: 1

   * - Network
     - Minimum mole-fraction ratio
     - Maximum mole-fraction ratio
     - Ti allocation at the largest response
   * - Ions, 537 species
     - 0.8932126
     - 0.9519298
     - 50.0872% to 47.6795%
   * - Neutral, 421 species
     - 0.8932126
     - 0.9519298
     - 50.0872% to 47.6795%

The highlighted allocation is at 1500 K, 100 bar and reference C/O. Across both
networks, Ti conservation is within 2.48e-9 relative and H/C/O within 2.58e-13;
ion charge is within 1.58e-14 and mass action within 1.37e-12 in log units.
Trace-element residuals reach 1.38e-6 and all-species tighter-control variation
1.38e-5 in ln mole fraction; these are distinct from target accuracy.

Relative to the older ``curated-cos-v1`` production archive, the cumulative
CN/PS/TiO2 update gives a TiO2 mole-fraction ratio of 0.8892665--0.9519298 in
the ion-inclusive grid. Comparing the new result to the earlier TiO2-only
candidate instead gives a ratio of 0.9930776--1.000000002. The additional
approximately 0.692% change comes from the CN/PS background, not from a failed
reproduction of the TiO2 potential. The old candidate's neutral comparison
remains unavailable and is not inferred from the ion-inclusive archive.

Scientific limits
-----------------

Ground-state RRHO omits anharmonicity, centrifugal distortion and a complete
excited-state manifold. The earlier equal-rotation A-state sensitivity of
+0.882% in K at 4000 K and bend interval of approximately +/-1.8% are selected
sensitivity calculations, not a full uncertainty budget. Historical JANAF energy
and entropy evaluation scales are retained; this thermal update does not reduce
them by assertion. A replacement D0 requires its own reaction-cycle evaluation.

Gas-only numerical conservation and independent-solver agreement do not prove
condensation, cloud-nucleation, opacity, transport or planetary atmospheric
accuracy. MCEG separates the adopted bounded thermal choice and numerical
execution from complete physical accuracy, which remains unknown. Missing
ancestral energy papers do not prevent this explicitly retained-energy thermal
adoption.

Reproduction
------------

Rebuild the packaged record with::

   python mceg/reviews/build_tio2_adoption.py

Run fresh production validation with::

   python mceg/reviews/curated_tio2_adoption.py \
       --output-dir results/tio2-adoption-new \
       --fastchem-executable /path/to/fastchem

Replay the offline evidence with::

   pytest tests/model_evidence/curated_tio2_adoption_test.py

The production packet is under
``tests/data/validation/curated_tio2_adoption/20261106``; the MCEG adoption
contract is ``mceg/contracts/curated_tio2_adoption.yaml``. Original source and
candidate packets remain immutable records of their earlier stage.

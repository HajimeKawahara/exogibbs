VO: modern energy and hot-state populations
===========================================

This 2026-11-03 investigation extends :doc:`gas_source_comparison` for
``O1V1`` (neutral VO), addressing Issue #308. **Retain the current curated
energy choice and investigate a modern thermal replacement.** Curated already
combines JANAF molecular thermal functions with Merriles (2020)
D0=6.545 +/- 0.002 eV and NASA neutral-atom thermal references. Reverting to
NASA would discard the stronger modern bond energy. Conversely, retaining
JANAF indefinitely is not justified by this review: normalized HyVO populations
suggest a substantial high-temperature thermal correction.

Why atmospheric priority matters
--------------------------------

The visible and near-infrared rovibronic bands in the primary
`HyVO study <https://doi.org/10.1093/mnras/stae542>`_ motivate prioritizing VO
as an atmospheric absorber even though vanadium is scarce. Improved line
positions, hyperfine splitting and dipole intensities address spectral accuracy;
they do not independently establish thermochemical equilibrium accuracy.
The archived gas-only grid lacks condensates, cold trapping, rainout, transport
and irradiation. Its VO abundance is not a predicted atmospheric detection or
an optical depth, nor does a change in K imply the same fractional abundance change.

The energy anchor is now checked beyond the abstract
----------------------------------------------------

`Merriles et al. (2020), 10.1063/5.0014006
<https://doi.org/10.1063/5.0014006>`_, reports the neutral VO predissociation
threshold as 52790 +/- 15 cm^-1, or 6.545 +/- 0.002 eV.
The author's accepted manuscript was inspected through the
`public rendered full text
<https://www.researchgate.net/publication/342901201_The_bond_dissociation_energy_of_VO_measured_by_resonant_three-photon_ionization_spectroscopy>`_,
including experimental controls and systematic discussion. PDF retrieval failed;
raw measurements were not reanalysed. This closes the earlier abstract-only
inspection gap in the current review, without rewriting historical adoption records.

A 150 ns delay separates competing ionization pathways; blocking either laser
removes the selected signal. OPO and dye calibration errors are below 5 and
1 cm^-1. Four excitation schemes fit the assigned 15 cm^-1 error, including
a 9 cm^-1 shift plausibly due to the initial fine-structure level. The authors
acknowledge that a dissociation barrier would make the threshold an upper bound,
but argue the dense state manifold and agreement across schemes make that unlikely.
This supports retaining the adopted energy, with its stated interpretation.
The derived cation energy, 6.053 eV, is a different reaction.

The quoted 0.002 eV alone corresponds to K factors 1.0156 at 1500 K and
1.00582 at 4000 K. These are energy-only sensitivities, not a confidence
interval for thermal functions, K, abundances or atmospheric models.

Native atoms and the current hybrid
------------------------------------

Fresh official `JANAF O-026 <https://janaf.nist.gov/tables/O-026.html>`_,
`V-005 <https://janaf.nist.gov/tables/V-005.html>`_ and
`O-001 <https://janaf.nist.gov/tables/O-001.html>`_ values reproduce the frozen
adoption inputs and broad source screen. VO's JANAF evaluation is December 1973.
NASA coefficients/headers come from the pinned adoption reference, citing
Gurvich (1982), part 1 p.62 and part 2 p.64; that original was not inspected.

For each source, H(0)=Hf(298.15)-[H(298.15)-H(0)] is first assigned in that
source's gauge, then the V and O atoms are balanced. This gives D0=631.171
kJ/mol for JANAF and 612.000 kJ/mol for NASA headers, approximately 6.5416 and
6.3429 eV. NASA's R=8.314510 convention is preserved when reconstructing its
dimensionless functions; JANAF uses R=8.31446261815324. A molecular Hf difference
alone is not a balanced reaction correction.

.. list-table:: Formation K ratios at 1 bar for V(g) + O(g) -> VO(g)
   :header-rows: 1

   * - T, K
     - Current curated / raw NASA
     - JANAF / NASA thermal, both at adopted D0
   * - 1500
     - 4.74282
     - 0.993116
   * - 2000
     - 3.16293
     - 0.979065
   * - 3000
     - 2.02930
     - 0.928600
   * - 4000
     - 1.53797
     - 0.855678

The roughly 0.20 eV energy update explains most of the low-temperature
contrast, while molecular thermal differences increasingly offset it at high T.
The native JANAF-minus-NASA atomic thermal contribution stays below 0.00032
in ln K at the six inspected nodes. The current hybrid deliberately uses NASA
atomic thermal references. Its actual polynomial, rather than the old FastChem
VO fit, reproduces the adopted JANAF/energy nodes and all saved VO potentials.
The production interval remains 1500-4000 K; 500 and 1000 K are source diagnostics.

Normalized modern partition diagnostic
---------------------------------------

Official HyVO and `VOMYT <https://doi.org/10.1093/mnras/stw1969>`_ partition
files were downloaded and hashed; selected exact rows are frozen. HyVO's
definition file is version 20231218. The molecular partition comparison uses::

   ln Q_JANAF = -(G-H0)/(R*T) - ln[(2*pi*m*k*T/h^2)^(3/2)*k*T/p0]
   Q_HyVO_thermochemical = Q_HyVO_download / 8
   K_proxy / K_current = (Q_HyVO_thermochemical / Q_JANAF)
                         * (66.93887412 / 66.9409)^(3/2)

The factor eight removes the 51V nuclear-spin multiplicity. Resolved HyVO
weights are 2F+1; their sum restores the unresolved 8*(2J+1). Multiplying
each resolved level by eight again, or using full molecular Q with
spin-excluded atomic functions, creates a spurious factor-eight shift.
The molecular masses are 66.93887412 u for 51V16O and 66.9409 u for the legacy
natural-element record; their translational correction is 0.999954605.

HyVO sets its zero at X4Sigma-, v=0, J=0.5, e parity, F=3. The first listed
F=0 state is not the zero. Exact alignment with the legacy fine/hyperfine and
adopted-D0 zero has not been rederived. The proxy holds D0 and NASA atomic
thermal functions fixed and assumes that alignment. It is not a fully
isotope-resolved reaction or an adopted replacement.

.. list-table:: Internal populations and conditional formation-constant change
   :header-rows: 1

   * - T, K
     - HyVO / JANAF internal Q
     - K proxy / current
     - Downloaded VOMYT / HyVO
   * - 1500
     - 1.00278
     - 1.00274
     - 0.999861
   * - 2000
     - 1.02504
     - 1.02499
     - 0.999866
   * - 3000
     - 1.15273
     - 1.15268
     - 1.00372
   * - 4000
     - 1.37980
     - 1.37974
     - 1.01788

HyVO includes 15 electronic states, energies through 45000 cm^-1 and F<=150;
the broader VOMYT limits are 50000 cm^-1 and J<=197.5. HyVO's 5400 K
completeness declaration corresponds to a 5% comparison criterion, not an
absolute physical error bound. At 4000 K, the paper's outer-level addition
increases HyVO Q by 1.44177%; the adjusted value is 0.25476% below its printed
VOMYT value. This is too small to explain the 38% JANAF difference.
Downloaded VOMYT differs from the paper's Table 3 by 0.08602% at 4000 K;
both versions are retained separately. No missing-level correction is adopted.

MCEG conclusion and concrete next step
---------------------------------------

The 75-case, 537-species archived ``curated-c2h-v1`` grid covers 1500-4000 K,
1e-6-100 bar and three C/O values. Its largest VO vanadium-budget fraction is
**0.588117 (58.81%)**, at 1500 K, 1 bar and C/O=1, despite a mole fraction of
only **8.01710e-9**. All cases converged; vanadium conservation and all VO
potentials are checked. This is a baseline footprint, not a new substitution
response or a physical validation of the atmosphere.

``mceg/contracts/vo_source_review.yaml`` separately records supported source
reconstruction, retained energy reasoning and normalized partition diagnostics;
complete thermal and atmospheric accuracy remain unknown. The next step is to
construct a continuous HyVO-based G/H/S/Cp candidate on 1500-4000 K, resolving
zero alignment and molecular/atomic isotope conventions and documenting the
high-level cutoff. Then perform controlled finite substitution in a fixed
network, checking conservation, mass action, absolute abundances and vanadium
allocation before adopting it. Missing every historical Gurvich original is
not a prerequisite to that modern candidate. Condensate and atmospheric
transport applicability are separate from the gas-source adoption decision.

No production data, preset identifiers or temperature bounds change here.
VO2 and the Gunina-dependent PR #318 are outside this review. The compact
packet is ``tests/data/validation/vo_source_review/20261103``; reproduce offline::

   PYTHONPATH=src python mceg/reviews/vo_source_review.py
   PYTHONPATH=src pytest tests/model_evidence/vo_source_review_test.py

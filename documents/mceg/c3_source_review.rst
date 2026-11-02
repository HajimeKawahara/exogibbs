C3 source investigation and carbon redistribution
=================================================

C3 is investigated after AlH because it holds up to 2.25410% of carbon in the
saved current-curated gas grid, at 2000 K, 1e-6 bar and C/O = 1.5. Its current
JANAF-derived fit is inherited and unreviewed. The priority is specific to
this finite gas-only grid, not a universal ranking of atmospheric importance.
The AlH investigation remains separate while its original-source checks await
access; this review does not depend on the unmerged AlH or PH2 changes.

The evidence favours the ATcT energy reference and a modern review of the
thermal functions. It does not justify replacing the complete C3 model with
raw NASA thermochemistry. Production coefficients, dataset identities and
historical snapshots remain unchanged.

Identity, energy and source attribution
---------------------------------------

The species is neutral gas-phase linear tricarbon in its ground electronic
state, X1Sigma_g+. The balanced comparison is 3 C(g) -> C3(g) at standard
pressure 1 bar. A cyclic C3 isomer is a different species; the ATcT records
for linear CCC and cyclic C(CC) must not be merged by molecular formula.

The JANAF table has Hf(298.15 K) = 820.064 kJ/mol. Its printed uncertainty is
about 17 kJ/mol. The NASA record cites Gurvich (1979), part 2 p. 23, and gives
839.948646 kJ/mol in its original header. Evaluating the polynomial with the
original NASA gas constant reproduces this header; the frozen comparison's
slightly different gas constant is recorded separately.

`ATcT version 1.172
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=201>`_
identifies linear CCC(g), with Hf(0 K) = 814.37 and
Hf(298.15 K) = 823.33 +/- 0.49 kJ/mol. Its quoted uncertainty is a 95% confidence
limit. The matching atomic-carbon values are 711.381 and 716.866 kJ/mol.
Comparison uses both molecular and atomic values from the same version;
it does not shift the molecular formation enthalpy while silently retaining
incompatible atomic energy references. The network provenance and shared
contributors are retained, and individual uncertainties are not combined
as if they were independent.

For the balanced formation reaction at 0 K, the enthalpies are -1319.773
kJ/mol (ATcT), -1322.099 (JANAF) and -1302.592815 (NASA headers). ATcT therefore
changes the JANAF reaction energy by +2.326 kJ/mol and the NASA reaction
energy by -17.180185 kJ/mol. This is evidence favouring a modern ATcT anchor,
closer to JANAF than NASA; it does not validate JANAF's thermal functions.
The full atomization reaction must also be distinguished from C3 -> C2 + C.

Printed and online JANAF C3 agree at seven inspected nodes. The formation
columns close against graphite within tabular rounding over 61 common positive
temperatures: maximum H error 0.002 kJ/mol and G error 0.0097 kJ/mol through
6000 K. The current fit reproduces JANAF over 26 nodes from 1500 to 4000 K
within 0.01104 in ln K, about 1.11%. Its reaction-Cp derivative differs from
the tabulated value by up to about 1.373 J/mol/K.

Energy alignment does not remove the thermal discrepancy
--------------------------------------------------------

Current-curated/NASA formation-K ratios are 2.63911 at 1500 K and 0.808994 at
3000 K. The ordering reverses with temperature. The JANAF-minus-NASA reaction
enthalpy difference at 298.15 K is about -19.8621 kJ/mol in the frozen numerical
convention. After aligning that anchor, JANAF/NASA K is still only 0.530905
at 1500 K and 0.366702 at 3000 K. Atomic-carbon contributions are small;
the report separates them from molecular energy, enthalpy-increment and entropy
terms. An energy-only shift cannot explain this source difference.

The printed JANAF note already includes the low bending frequency near
63.5 cm^-1. It does not simply omit the soft bend. For higher bending levels,
it introduces an empirically stiffened potential to fit older high-temperature
entropy information, with an explicitly arbitrary estimate beyond measured
levels. This treatment is a material candidate explanation for the thermal
difference. An old Irwin partition-function discrepancy discussed in recent
literature concerns different source ancestry and is not attributed directly
to JANAF here.

NASA TP 1141 (Jones, 1978) discusses the same older entropy problem and quotes
later Wachi-Gilmartin (1972) measurements. At 2400 K the quoted entropy is about
340.87 J/mol/K before a conditional pressure conversion, closer to NASA's
339.88 than JANAF's 328.16 or the published AtLast table's 334.47. However,
the second quoted value, about 346.90 J/mol/K, is assigned to 2600 K in the
authors' conference abstract and 2700 K by Jones. Both readings are preserved;
neither temperature is silently selected as a verified experimental point.
The journal article, uncertainty and standard pressure remain unverified.
The Jones/Lee-Sanborn model is also not established as identical to the later
Gurvich/NASA source. This historical comparison is a lead for evaluating larger
thermal functions, not a completed independent validation.

The `2024 ExoMol AtLast model <https://doi.org/10.1093/mnras/stae2425>`_
provides an independent ground-state rovibrational
comparison with an anharmonic bending treatment and empirical spectroscopic
constraints. It includes levels up to 20000 cm^-1, J = 155 and bending angular
momentum l = 12 for 12C3. Its authors explicitly describe the partition function
as a lower limit because excited electronic states are omitted. Numerical
convergence of a particular state sum does not certify a complete thermal model.

The archived 12C3 definition file declares a nuclear-spin degeneracy of 6,
whereas the original paper gives zero nuclear spin for 12C and no multiplicative
spin correction for 12C3. The archived partition function and low-temperature
state checks are treated separately from this inconsistent metadata; dividing
the partition function by 6 would not be justified. All 2,166,503 downloaded
state weights are 2J+1, consistent with a spin factor of 1. Original bytes are
retained.

The full state sum also differs from the published partition file. Their Q
ratio is about 0.8578 at 300 K, 0.9802 at 1500 K and 0.9962 at 3000 K.
Using the auxiliary calculated-energy column does not resolve the discrepancy.
This is an observed inconsistency whose historical cause is undetermined;
it must not be silently repaired or described as a known version mix-up.
The optional full-state calculation has its own source digest and reproduction
command. Ordinary offline tests use the compact saved diagnostics and do not
download or re-sum the large external state file.

Under the recorded legacy-mass and source-gas-constant conventions,
published-AtLast-table/JANAF Q ratios are about 1.531 at 1500 K and 1.799 at
3000 K; published-table/NASA ratios are about 0.836 and 0.669. The directly
recomputed state-sum/JANAF ratios are about 1.500 and 1.792. Thus even the
modern ground-state sum exceeds JANAF materially, while it does not reproduce
NASA's larger sum.
This supports re-evaluating the old thermal treatment, rather than declaring
either full source physically accurate from their mutual comparison.

Heat capacity is checked independently of Q's value. Differentiation of the
AtLast table gives Cp about 50.64, 47.19 and 40.55 J/mol/K at 1500, 3000 and
4000 K, respectively. Direct energy-variance moments of the complete downloaded
state file reproduce the decline, giving about 50.44, 47.07 and 40.48 J/mol/K.
The decline is therefore not simply a finite-difference artifact. These
finite-model values must be assessed for state
coverage and caloric convergence; a stable finite-difference step is not a
physical uncertainty estimate. The source packet records the derivative,
normalization and state-coverage diagnostics and their limits.

Controlled raw-NASA substitution
--------------------------------

The diagnostic replaces only C3 in ``curated-c2h-v1`` with the frozen NASA
C3-3C reaction, under the separate identity
``curated-c2h-v1+raw-nasa-c3-diagnostic``. It preserves the 537-species
ion-inclusive network, formula matrix and each pair's elemental budget.
Only C/H changes between the three C/O settings, before normalization.

There are 75 pairs: temperatures 1500, 2000, 2500, 3000 and 4000 K; pressures
1e-6, 1e-4, 0.01, 1 and 100 bar; C/O = 0.5888436553555908, 1 and 1.5.
Both endpoints are cold-solved at tolerances 1e-13 and 5e-14. All 300 solutions
pass convergence, conservation and mass-action checks. Full species amounts,
logarithms, mole fractions, differences, thermochemistry and C/O/N allocations
are saved, together with the formula matrix and budgets.

.. list-table:: C/O = 1.5, 2000 K, 1e-6 bar
   :header-rows: 1

   * - Species
     - Current mole fraction
     - Raw-NASA C3 probe
     - Relative change
   * - C3
     - 6.08215e-6
     - 4.19707e-6
     - -30.99%
   * - C3H
     - 3.14160e-5
     - 3.24400e-5
     - +3.259%
   * - C2H2
     - 4.67992e-5
     - 4.78106e-5
     - +2.161%
   * - HCN
     - 2.38136e-5
     - 2.40125e-5
     - +0.8355%

This condition has both the largest current C3 carbon share and the largest
absolute C3 mole-fraction change, -1.88508e-6. The share of carbon in C3 falls
from 2.25410% to 1.55547%. The largest recipient is the fixed network's C3H
entry (``C3H1``), whose carbon share rises from 11.64305% to 12.02252%; C2H2
rises from 11.56281% to 11.81267%. The response must not be described as transfer
only into C2H2 and HCN. CO changes by +1.16061e-9, about +0.0002151%.
All carbon-bearing species are retained in the partition report.

Maximum tolerance differences are below 9.21e-14 in x and 2.37e-5 in ln(x).
At this same highlighted condition, both endpoints agree with standalone
FastChem within 9.78e-7 dex over all 537 species, using identical constants
and budgets. This verifies a numerical sensitivity, not a chosen replacement
or atmospheric accuracy. The decrease at this condition is not universal:
the source K ratio reverses with temperature.

Scientific assessment and follow-up
-----------------------------------

ATcT provides a better-supported energy anchor than choosing between the two
legacy central values. The thermal comparison identifies a substantive weakness
in treating the old JANAF fit as validated, but raw NASA is not established
as the preferred complete replacement. A candidate should combine a justified
balanced energy with separately assessed thermal functions, including the
soft bending manifold, high-energy coverage and electronic contributions.
The AtLast table should not be adopted across 1500--4000 K merely because it
is newer or because low-lying spectral lines agree with experiment.

The current source is retained operationally while that candidate is assessed;
this is not an endorsement of its physical accuracy. The raw-NASA probe is
neither an uncertainty interval nor the predicted response to a future
ATcT/modern-thermal correction. Such a candidate needs its own execution record.
Condensation, nonideality, nonequilibrium chemistry and spectra are outside
this gas-only study.

C3H warrants a separate identity/source review: its unchanged inherited fit
becomes the largest recipient in this test, and it already holds over 11% of
carbon at the highlighted point. That observation motivates a follow-up;
it does not validate its identity or thermochemistry.

Evidence and reproduction
-------------------------

Source and sensitivity packets are saved under
``tests/data/validation/c3_source_review/20261102`` and
``tests/data/validation/c3_equilibrium_impact/20261102``. Their READMEs give
reproduction commands. Original-source access, selected input/output hashes,
execution environments and scientific limits are recorded independently.

``mceg/contracts/c3_source_review.yaml`` separates source attribution,
conditional source-selection reasoning, finite substitution and the unresolved
accuracy of a complete thermal model or atmosphere. Ordinary checks require
neither network access, MCEG nor a FastChem executable::

   PYTHONPATH=src pytest tests/model_evidence/c3_source_review_test.py \
     tests/model_evidence/c3_equilibrium_impact_test.py

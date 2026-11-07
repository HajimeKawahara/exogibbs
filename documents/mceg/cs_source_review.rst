CS: thermal audit and a newer balanced energy
=============================================

This 2026-11-05 investigation addresses Issues #308 and #309 for neutral
``C1S1``. **Keep the current thermal functions; evaluate an ATcT 1.228
energy-only candidate next.** Production remains unchanged in this source
review. Retaining its older energy pending that calculation is not a claim
that ATcT 1.172 is scientifically preferable to the newer evaluation.

CS is important even where molecular thermal differences are small. In the
75 saved ``curated-h2o-v1`` cases, CS carries up to **93.4736% of sulfur**
at 2000 K, 1e-6 bar and C/O=1.5. The corresponding mole fraction is
1.35770e-5. All saved CS potentials reproduce to rounding, and the sulfur
budget residual is below 3e-12. These are existing equilibrium endpoints,
not a new abundance response or a validation against a planetary atmosphere.

Energy balance
--------------

The current construction is NASA thermal functions plus the balanced
`ATcT 1.172 <https://atct.anl.gov/Thermochemical%20Data/version%201.172/index.php>`_
ground-state atomization energy:

.. math::

   D_0 = H_f^0(\mathrm{C}) + H_f^0(\mathrm{S}) - H_f^0(\mathrm{CS})
       = 711.381 + 276.830 - 278.900 = 709.311\ \mathrm{kJ/mol}.

Native NASA and JANAF give 710.814980 and 708.837 kJ/mol respectively.
NASA's historical R=8.314510 is retained in its dimensionless thermal
functions and energy anchor; modern R=8.31446261815324 is used for state
sums and common-unit caloric comparisons. The hybrid shift is
-180.400027 K, producing curated/NASA formation-K ratios 0.886684 at
1500 K and 0.941639 at 3000 K. This difference is an intentional energy
choice, not evidence of a missing rotational factor.

The inspected official ATcT 1.228 entries for
`C <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=21>`_,
`S <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=138>`_
and `CS <https://atct.anl.gov/Thermochemical%20Data/version%201.228/species/?species_number=1425>`_
give 711.370 + 276.85 - 278.53 = **709.690 kJ/mol**. All three values come
from the same network version. The +0.379 kJ/mol change would raise
formation K by 3.086% at 1500 K and 1.146% at 4000 K, with thermal functions
held fixed. This is a calculated K sensitivity, not a computed abundance
change. The CS entry's reported 0.52 kJ/mol uncertainty is not a reaction
uncertainty; covariance was not propagated. The version change is not
established as a statistically resolved discrepancy.

Both atomization reactions use ground C(3P0), S(3P2) and molecular X1Sigma+
v=0,J=0 zeros. Atomic excited-level populations remain in the NASA atomic
thermal functions. Neither a PEC well depth De nor a single excited
dissociation asymptote can replace the balanced D0 without conversion.
The legacy NASA header cites Prinslow 1991 and Gurvich 1991; those originals
were not reread. This does not prevent merging the scoped source audit.

What the current JnK distribution supports
------------------------------------------

`Paulose et al. (2015), DOI 10.1093/mnras/stv1543
<https://doi.org/10.1093/mnras/stv1543>`_ was inspected in full. It covers
the ground electronic state and claims at least 95% partition convergence
at 3000 K; it does not establish 4000 K accuracy. The fixed PEC De of
59300 cm-1 is not adopted as thermochemical D0.

All **11497 distributed states** were summed, with actual J<=258 and v<=49.
For 12C32S, both nuclear spins are zero and gtot=2J+1. The energy origin is
the first state at zero; no additional spin or symmetry division applies.
The metadata maximum energy is 30000 cm-1, whereas the downloaded states
reach **66286.878662 cm-1**. Actual rows define this numerical audit.
There are 441 states above the adopted D0. Their removal changes Q at
4000 K by 5.62e-9 and Cp by 1.87e-5 J/mol/K. They are not silently treated
as independently verified bound levels.

The official PF ends at 3000 K. Direct sums reproduce its sampled values
within 0.6 ppm. Its Q(3000)=5741.4663 differs from the paper's Table 8 value
5705.80. The readme says updated March 2016, but the exact cause of the
paper/distribution difference is unestablished. Values above 3000 K below
are direct finite-list diagnostics, not extrapolated published PF values.

.. list-table:: Natural-mass comparison at fixed D0 and fixed atomic thermal functions
   :header-rows: 1

   * - T (K)
     - JnK Q / NASA-derived Q
     - JnK Cp (J/mol/K)
     - NASA Cp, common R (J/mol/K)
   * - 1500
     - 0.999926
     - 36.7645
     - 36.7395
   * - 3000
     - 1.000543
     - 37.8597
     - 37.8232
   * - 4000
     - 1.000915
     - 38.2639
     - 38.5696

The internal-state sum uses the representative isotope with NASA's natural
molecular translation mass, 44.0757 u. The separate 43.972071 u isotope-mass
proxy is recorded; neither is a full isotope-resolved equilibrium reaction.
JANAF and Barklem--Collet comparisons are also retained, without treating
shared historical spectroscopy as independent physical validation.

At 4000 K, J>=240 and v>=45 carry only 3.20e-7 and 1.98e-7 of the
distributed Q. These boundary checks do not bound every omitted state.
The close Q agreement also does not remove the 0.306 J/mol/K Cp difference.
A deliberately coarse sensitivity using five low electronic terms from
`NIST's evaluated constants <https://webbook.nist.gov/cgi/cbook.cgi?ID=C2944050&Mask=1000>`_
adds about 0.305 J/mol/K of Cp at 4000 K while changing Q by only 0.035%.
It assumes identical rovibrational sums and uses Te minima without zero-point
or spin-orbit corrections. This illustrates why electronic excitation matters
more to derivatives than Q; it is neither an adopted correction nor a proof
that NASA contains exactly these contributions.

Newer papers and the practical next step
----------------------------------------

The full `Li et al. (2022) paper, 10.1088/1674-1056/ac834f
<https://doi.org/10.1088/1674-1056/ac834f>`_ discusses twelve electronic
potentials. Section 4.1 nevertheless constructs its plotted Q from only
the lowest three singlet states. It does not supply the complete triplet-
inclusive thermal replacement needed here.

`Coxon and Hajigeorgiou (2023), 10.1016/j.jms.2023.111861
<https://doi.org/10.1016/j.jms.2023.111861>`_ improves the ground-state
potential using 6759 lines. Its abstract/introduction were inspected;
the full text and supplementary potential parameters were not obtained.
The full `Olivares-Pilon and Turbiner v2 preprint
<https://doi.org/10.48550/arXiv.2409.14875>`_ predicts additional high states
and discusses near-threshold model dependence. State counts alone do not
determine their thermal contribution.

The evidence supports retaining the existing thermal functions for this
review. **The next useful calculation is a balanced ATcT 1.228 energy-only
candidate and fixed-network sulfur redistribution test.** That step does
not require obtaining every legacy original. A later precision thermal
replacement would require converged Q, H, S and Cp with consistent electronic
zeros; the 2023 potential/supplement could help that calculation.

The offline packet and ``mceg/contracts/cs_source_review.yaml`` distinguish
verified construction, normalized numerical comparisons, scoped retention
and candidate priority, and unknown complete physical accuracy. No runtime
coefficient, production receipt or curated version is modified.

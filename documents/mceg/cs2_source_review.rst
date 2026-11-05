CS2: balanced energy update and partition-function crossover
============================================================

Decision
--------

Retain current NASA thermal functions provisionally and prioritize a balanced
ATcT 1.228 energy-only candidate. No curated coefficients are changed.
The current ``C1S2`` column combines NASA molecular and atomic thermal
functions with ATcT 1.172 energy. Its difference from native NASA is exactly
an energy shift, not a heat-capacity change.

Current balanced D0 is 1149.161 kJ/mol, from the same-version
C=711.381, S=276.830 and CS2=115.880 kJ/mol Hf(0). The
`ATcT 1.228 network
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
gives C=711.370, S=276.85 and CS2=115.17, so D0=1149.900 kJ/mol.
The 0.739 kJ/mol increase alone multiplies current formation K by
1.061045/1.030070/1.022469 at 1500/3000/4000 K. Native NASA D0 is
1145.133478 kJ/mol and native JANAF D0 is 1144.524. These balanced
comparisons preserve source R; network shifts are not independent measured
errors, and reaction covariance has not been supplied.

Thermal comparison
------------------

Current/NASA K is 1.381896/1.175541/1.128959 at 1500/3000/4000 K.
Using common D0 and atom functions, JANAF/NASA molecular thermal ratios are
1.035852/1.072196/1.094119; common-R Cp differences are
+0.57126/+0.93563/+0.41810 J/mol/K. NASA cites Gurvich (1991), while
JANAF C-099 is evaluated in December 1976. Those historical evaluations
have not been reconstructed from all original literature.

`TIPS2025 <https://doi.org/10.1016/j.jqsrt.2025.109568>`_ updates CS2 using
Ames term values and a symmetry-aware triple product approximation. For
12C32S2 it switches at 2098 K. Nuclear spins are zero; symmetry-forbidden
levels are already accounted for, so there is no further factor-of-two
division. Ground-relative Q/NASA at 1500/3000/4000 K is
0.991007/0.962835/0.947107. Changing molecular translation from natural
76.1407 u to 75.944142 u gives fixed-atom K proxies
0.987172/0.959109/0.943442. This does not form an isotopic mixture.

The first 10 K grid node beyond the crossover, 2100 K, is 1.0123 percent
above continuation of the previous local log-Q slope. Such a join cannot
be adopted as a differentiable thermodynamic function without treatment.
Five-point derivatives at 10/20 K step give Cp=54.576/54.683 J/mol/K at
4000 K, versus NASA 64.489. The 10 K result has a conservative rounding-only
bound of 0.443 J/mol/K. This substantial caloric disagreement persists
beyond table rounding; it is not evidence that the newer product
approximation is automatically preferable. Missing-state and approximate
rotation/vibration contributions remain to be evaluated.

`Ames-296K <https://doi.org/10.3847/1538-4365/ad3809>`_ provides useful
spectroscopic term values, but the inspected NASA abstract establishes
room-temperature spectral validation, not a hot thermodynamic benchmark.
The full `2025 MARVEL study <https://doi.org/10.1016/j.jms.2025.111998>`_
identifies 4279 empirical levels and high-bending discrepancies against
Ames. Sparse observed levels cannot be summed as a complete high-temperature
partition function. These are available follow-up checks, not a demand for
unlimited additional literature before merging this review.

Footprint and evidence boundary
-------------------------------

The archived ``curated-oh-v1`` 75-case, 537-species gas grid reproduces all
CS2 potentials and converged endpoints. Maximum sulfur fraction is
5.84523e-6 (case 68), including the factor of two sulfur atoms per molecule.
It is an existing footprint, not a new abundance response or a statement
about every planetary atmosphere.

Pinned HAPI arrays, exact source references, finite-difference controls and
an offline replay are supplied. The pinned TIPS v1.2 release is distinguished
from the July 2026 v3 label correction. MCEG records supported comparisons
and energy-candidate priority, while full physical accuracy and historical
source completeness remain unknown. A thermal replacement needs a smooth,
calorically assessed model; an energy-only candidate can be investigated
independently with the already inspected same-version ATcT values.

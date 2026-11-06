KCN bounded thermal candidate and substitution
==============================================

The evaluated T-shaped three-mode RRHO construction is a finite sensitivity,
not a recommended production thermal replacement. It removes the obsolete
linear mode count, but the primary papers show large-amplitude motion at
energies populated throughout the tested temperature range. Retain curated
while completing the specific coupled-state calculation described below.
The original source-review packet and runtime remain unchanged.

New primary inputs and applicability
------------------------------------

The complete author copies of Tennyson and van der Avoird,
DOIs `10.1080/00268978200101101 <https://doi.org/10.1080/00268978200101101>`_
and `10.1063/1.442966 <https://doi.org/10.1063/1.442966>`_, are now inspected.
The latter close-coupling calculation at CN = 1.171 Angstrom gives the two
low fundamentals 293.0 and 114.7 cm^-1. Its reported gas CN stretch is
2158 cm^-1. These three values are harmonic surrogates in the main sensitivity;
they are not claimed to be exact small-amplitude harmonic frequencies.
The earlier Watson-model results 302.7 and 119.7 cm^-1 provide a selected
control with the same CN stretch. Matrix-isolation values 2050, 288 and
139 cm^-1 are not relabelled as gas measurements.

The paper identifies a 503.9 cm^-1 inversion barrier and strong rotation--
vibration coupling, with explicit failure of small-amplitude expansions near
and above that barrier. At 1500 K, kT/(hc) is about 1043 cm^-1. The low-lying
states in these papers therefore do not establish a complete hot partition
function. The energy relative to the ionic K + CN-minus asymptote in the
variational paper is not the neutral-atom D0 and is not inserted as a new
formation anchor. The modern nonlinear potential follow-up
`10.1063/1.4830102 <https://doi.org/10.1063/1.4830102>`_ does not turn this
three-oscillator calculation into a converged coupled-state sum.

Measured rotation and the physical gauge
----------------------------------------

The measured ground-vibrational principal-isotope rotational constants remain
those of `10.1016/j.jms.2011.12.005 <https://doi.org/10.1016/j.jms.2011.12.005>`_.
The frozen definition records A/B/C in MHz. The nonlinear rotor has symmetry
number 1, ground electronic weight 1, and exactly three oscillators. The old
linear JANAF proxy has four oscillators, including the two degenerate bending
components. Both use the same original natural mean translational mass and
1 bar pressure; no extra nuclear-spin, isotope-mixing or symmetry factor is
applied. Principal-isotope rotation with mean translation is an explicit
approximation, not a full isotope-resolved model.

The old linear RRHO construction independently reproduces the printed JANAF
calorics within 3 J/mol in H and 0.003 J/(mol K) in S/Cp on the inspected nodes.
The candidate uses the potential identity::

    Gnew = Gcurrent + phi(T-shaped RRHO) - phi(original linear RRHO)

Here phi is molecular G-H0. This preserves the full current-fit residual
relative to that reconstructed original proxy and leaves the atomic gauge
unchanged. Physical JANAF D0 = 1192383 J/mol is retained.
The empirical fit intercept is recorded separately and is not reinterpreted
as a dissociation energy. No new ZPE, H298 anchor or energy adjustment is added.
One potential supplies analytic G/H/S/Cp; 2752 dense/off-grid scalar/JAX
checks cover both the main model and selected low-mode control.

Finite substitution and numerical limits
----------------------------------------

The actual candidate has 150 fresh cold endpoints: 75 ion-inclusive cases
with 537 species and 75 neutral cases with 421 species. The curated-ch2-v1
baseline is reused only after its hashes, actual named-factory potentials,
formula matrix, species, budgets and conditions agree. There are 12 tighter
cold solves, 10 path nodes, 4 selected alternative-mode solves and 4 independent
FastChem runs on two selected ion-inclusive conditions. Every element, charge
and formation closure is checked, and only the target potential changes.

In the ion-inclusive network, the target mole-fraction ratio is
8.00917864--11.73533163. Maximum K allocation changes
from 0.269884828% to 3.077652318% at 1500 K,
0.01 bar and C/O = 1.5. The corresponding mole fraction is
5.441088195e-10 -> 6.204786625e-09. These are solved responses of the stated
sensitivity, not measured abundance errors or evidence that the new thermal
approximation is physically correct. The full partition ratios differ from
both the earlier rotation-only diagnostic and the equilibrium abundance ratios.
The alternate-mode response is archived independently, not averaged into a
preferred fit or interpreted as a complete uncertainty interval.

The largest independent FastChem discrepancy is 1.46e-06 dex. Target log-mole
fractions remain stable within 1.2e-13 under the tighter solve; the largest
all-species change is about 3.4e-6 in a trace log-mole fraction, within the
explicit control threshold. Same-model Gibbs exchanges resolve above the
recorded diagnostic scale in 48/75 ion-inclusive and 57/75 neutral cases.
The remaining comparisons are unresolved, not failed physical inequalities.
This scale is not a rigorous optimization error bound or a source ranking.

Decision and finite follow-up
-----------------------------

Do not adopt this RRHO sensitivity. The large response makes a complete
nonlinear thermal treatment worthwhile, but a known low barrier cannot be
ignored merely because the numerical solve converges. Use a converged rotational/large-amplitude state calculation on an inspected KCN potential (including DOI10.1063/1.4830102 as a modern potential follow-up), with a justified high-energy cutoff and direct thermal checks. Do not substitute a rotation-only factor or isolated ten-level list for a complete Q.
The retained source energy has its own historical uncertainty and is not
improved by this thermal exercise. Condensation, photochemistry, complete
reaction covariance and observational atmospheric accuracy remain outside
this fixed ideal-gas comparison. MCEG distinguishes supported numerical and
scoped decision claims from unknown full physical accuracy and production
adoption; unknown is not a source-review merge blocker.

Offline evidence
----------------

The candidate, source facts, validation and independent execution packets are
pinned by ``mceg/contracts/kcn_candidate.yaml``. The replay test verifies
original input hashes through the existing historical resolver, direct output
hashes, archived baseline identity, target-only changes and all finite controls.
No network or FastChem installation is required for the offline replay.

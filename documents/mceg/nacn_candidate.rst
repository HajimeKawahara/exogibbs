NaCN bounded thermal candidate and substitution
===============================================

The evaluated T-shaped three-mode RRHO construction is a finite sensitivity,
not a recommended production thermal replacement. It removes the obsolete
linear mode count, but the primary papers show large-amplitude motion at
energies populated throughout the tested temperature range. Retain curated
while completing the specific coupled-state calculation described below.
The original source-review packet and runtime remain unchanged.

New primary inputs and applicability
------------------------------------

The full primary Senent, Dumouchel and Lique paper (2012,
DOI `10.1111/j.1365-2966.2011.20132.x <https://doi.org/10.1111/j.1365-2966.2011.20132.x>`_)
and its Table 1 image are now inspected. Table 1 gives harmonic modes
2061, 373 and 181 cm^-1 for the T-shaped minimum. These three modes define
the main sensitivity. Table 2a fundamentals 2031, 173 and 460 cm^-1 define
an alternate harmonic-surrogate control, not an infinite anharmonic state sum.
The large x22 = -50.592 and x23 = 188.568 cm^-1 are reasons not to extrapolate
a truncated Dunham polynomial to arbitrarily high quantum numbers.

The same potential has linear NaNC and NaCN minima at 774 and 944 cm^-1,
with transition configurations at 787 and 1380 cm^-1 above the T-shaped
minimum. They are connected regions of one molecular potential; counting them
as independent extra species would require a separate, justified construction.
At 1500 K, kT/(hc) is already about 1043 cm^-1. Therefore even the coldest
production-grid point does not validate a single-well harmonic approximation.
No interval of this 1500--4000 K calculation is promoted to an author-validated
high-temperature thermal model.

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
unchanged. Physical JANAF D0 = 1195582 J/mol is retained.
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
5.62061812--8.20166965. Maximum Na allocation changes
from 0.240904692% to 1.941567457% at 1500 K,
0.01 bar and C/O = 1.5. The corresponding mole fraction is
6.8604492e-09 -> 5.529168016e-08. These are solved responses of the stated
sensitivity, not measured abundance errors or evidence that the new thermal
approximation is physically correct. The full partition ratios differ from
both the earlier rotation-only diagnostic and the equilibrium abundance ratios.
The alternate-mode response is archived independently, not averaged into a
preferred fit or interpreted as a complete uncertainty interval.

The largest independent FastChem discrepancy is 1.44e-06 dex. Target log-mole
fractions remain stable within 1.2e-13 under the tighter solve; the largest
all-species change is about 3.4e-6 in a trace log-mole fraction, within the
explicit control threshold. Same-model Gibbs exchanges resolve above the
recorded diagnostic scale in 53/75 ion-inclusive and 58/75 neutral cases.
The remaining comparisons are unresolved, not failed physical inequalities.
This scale is not a rigorous optimization error bound or a source ranking.

Decision and finite follow-up
-----------------------------

Do not adopt this RRHO sensitivity. The large response makes a complete
nonlinear thermal treatment worthwhile, but a known low barrier cannot be
ignored merely because the numerical solve converges. Obtain or compute a converged coupled rotational/large-amplitude state sum on the inspected2012potential, including its connected minima, with state weights and a finite dissociation prescription. Validate against measured spectra and GHSCp before a new thermal adoption decision.
The retained source energy has its own historical uncertainty and is not
improved by this thermal exercise. Condensation, photochemistry, complete
reaction covariance and observational atmospheric accuracy remain outside
this fixed ideal-gas comparison. MCEG distinguishes supported numerical and
scoped decision claims from unknown full physical accuracy and production
adoption; unknown is not a source-review merge blocker.

Offline evidence
----------------

The candidate, source facts, validation and independent execution packets are
pinned by ``mceg/contracts/nacn_candidate.yaml``. The replay test verifies
original input hashes through the existing historical resolver, direct output
hashes, archived baseline identity, target-only changes and all finite controls.
No network or FastChem installation is required for the offline replay.

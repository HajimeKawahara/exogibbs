AlH source investigation and aluminium redistribution
=====================================================

AlH is investigated next because it is a major aluminium carrier in the saved
current-curated grid and its adopted energy is explicitly provisional. At
2500 K, 100 bar and C/O = 1.5 it holds about 51% of aluminium. Among the remaining
P1 screening candidates, PO2 has the largest elemental share, but holds at most
0.0744% of P in this grid; NaOH dimer, N3 and S5 hold still smaller shares.
These finite-grid results motivate prioritizing AlH over the table's colour
order. They do not establish a universal atmospheric importance ranking.

The baseline is ``curated-c2h-v1``. This investigation has no dependency on the
unmerged PH2 review and does not change production coefficients, dataset
identities or historical evidence.

What the existing decision leaves unresolved
--------------------------------------------

The identified species is neutral gas-phase diatomic AlH, not alane (AlH3), an
ion or a condensed hydride. The comparison reaction is Al(g) + H(g) -> AlH(g)
at standard pressure 1 bar. Its ground electronic state is X1Sigma+.

The existing decision is ``retain_provisionally``: JANAF and NASA have different
energy thresholds, and the previously inspected Barklem-Collet value of
2.949 +/- 0.130 eV does not by itself distinguish them. The requested follow-up
is inspection of the primary dissociation measurement and caloric properties.

JANAF's 1963 AlH evaluation gives Hf(298.15 K) = 259.408 kJ/mol, with a printed
uncertainty of about 20 kJ/mol. It adopts Gaydon's 1953 estimate of
D0 = 2.9 +/- 0.2 eV. The NASA record
cites Gurvich (1996a), part 1 p. 139 and part 2 p. 106, and gives a header value
of 249.250804 kJ/mol. This roughly 10 kJ/mol difference is the principal source
of the formation-equilibrium discrepancy. The numerical audit separates
molecular energy, atomic references, enthalpy increments and entropy terms.

The balanced JANAF-minus-NASA reaction enthalpy at 298.15 K is about
10.4563 kJ/mol. After aligning that anchor, JANAF/NASA formation-K ratios are
0.999840 at 1500 K and 0.987108 at 3000 K. Atomic-reference contributions to
the Gibbs difference are about 0.29 kJ/mol at those temperatures. The current
FastChem fit agrees with JANAF H/S reconstruction within 0.00414 in ln K
(about 0.415%) across 26 nodes from 1500 to 4000 K.

Printed and online JANAF AlH agree at the six inspected nodes. The online
formation columns also close with Al(ref) + 0.5 H2 within tabular rounding;
Al reference phase changes are retained. There is no corresponding evidence
of the printed/online mismatch found for PH2.

Dissociation energy definitions matter
--------------------------------------

De measures dissociation from the bottom of a potential well; D0 measures it
from the lowest vibrational level. They differ by the zero-point energy.
Atomic fine-structure reference choices must also agree. A value labelled De
cannot be used as D0 without those conversions.

The supplied full text of Baltayan and Nedelec (1979),
`DOI 10.1063/1.437749 <https://doi.org/10.1063/1.437749>`_, resolves an important
metadata discrepancy. Both its printed abstract (p. 2399) and Section IV
(p. 2401) give De(X) = 3.16 +/- 0.01 eV. The previously inspected
publisher/Crossref abstract says 3.17 eV; the printed article takes precedence.
Later quotations of 3.16 eV as De agree with it, whereas labelling that value
D0 remains incorrect.

Section IV places the X-state dissociation limit 100 cm^-1 below the A-state
limit for Al(2P) fine structure; Section V identifies Al(2P) + H(2S) ground
atomic terms. The lower X asymptote is consistent with the lowest Al fine-
structure level, although the article does not print J labels. The stated
fine-structure offset is already included in De(X): subtracting a further
term-center correction of about 0.0093 eV would not follow the paper's
convention. Its approximate 100 cm^-1 offset is not silently replaced by the
modern 112.061 cm^-1 splitting.

The energy is inferred from a reconstructed potential, combining laser
lifetimes, emission intensity deficits and diffuse-line widths with a
semiclassical tunnelling model. Spectroscopic RKR and Morse segments constrain
the inner potential, while the outer barrier and tail supply the dissociation
limit. The main stated errors arise from high-J lifetimes and roughly
50 cm^-1 uncertainty in high-J level energies. Figure 4 excludes the latter
from its plotted error bars. The paper does not define the confidence level
of its +/-0.01 eV estimate, so it is not a calibrated statistical error bar.
The local PDF hash and reading locations are retained in the evidence packet;
the full article and extracted text are not redistributed.

The supplied `Szajna et al. (2011) article
<https://doi.org/10.12693/APhysPolA.120.417>`_ now supplies the original
vibrational constants, beyond the earlier rounded two-term quotation.
Table IV (p. 421) gives the signed coefficients Y10 through Y50 as
1682.37499, -29.05111, 0.247654, -0.0014152 and -0.00013 cm^-1.
Evaluating their five-term polynomial at v=0 gives 833.9555817 cm^-1,
or 0.1033973143 eV, when the potential-zero/Y00 correction is set to zero.
Together with the printed De=3.16 eV this gives diagnostic D0=3.0566026857 eV.
The k=3--5 contribution is only 0.00000383 eV relative to the same first two
coefficients. It cannot explain the approximately 0.108 eV JANAF/NASA gap.

The polynomial reproduces the printed Table II v=0--5 relative terms within
0.0003 cm^-1, checking the transcribed signs and powers. This is a check of
the same fitted low-level spectroscopy, not independent validation of an
absolute potential minimum. Y50 was fixed to White et al. (1993); neither its
uncertainty nor the coefficient covariance is given. Individual coefficient
errors cannot supply a complete zero-point or D0 confidence interval.
Higher terms and the potential-zero convention remain limitations, and the
polynomial is not extrapolated to dissociation. The Table IV symbol De is a
centrifugal-distortion constant, distinct from dissociation De in eV.

ExoMol partition functions require a separate convention check: the full
nuclear-spin weight for 27AlH is 12. Comparing these directly with a
thermochemical partition function that omits nuclear-spin degeneracy would
introduce a spurious entropy offset. A line list fitted to measured low-lying
levels also does not, by that fact alone, determine the dissociation asymptote.

The archived AloHa partition function, divided by 12, is compared with internal
partition functions reconstructed from JANAF and NASA H/S under explicit
mass, standard-pressure and source-gas-constant conventions. At 1500, 3000
and 4000 K, ExoMol/JANAF ratios are 1.00076, 1.00983 and 1.02069;
ExoMol/NASA ratios are 1.00033, 0.99662 and 0.97742. Thus both source partition
functions are within about 2.3% at these nodes.

That agreement does not imply equally small heat-capacity differences.
At 4000 K the molecular Cp values are 39.844 J/mol/K (JANAF) and about
45.560 J/mol/K (NASA). A five-point differentiation of the ExoMol Q table gives
about 40.603 J/mol/K. Changing the finite-difference step between 10, 20 and
40 K changes the saved 1500--4000 K diagnostic values by less than 0.01 J/mol/K.
This is a derivative-stability check, not a physical uncertainty bound.
The ExoMol model includes the X/A states and quasi-bound levels, so omitted
electronic states and their treatment still limit the physical interpretation.
The separate derivative audit of the current fit also distinguishes its
approximate caloric derivatives from the tabulated JANAF functions: its
reaction-Cp discrepancy reaches about 0.70 J/mol/K over the 26 audited nodes.

The supplied `AloHa publisher article <https://doi.org/10.1093/mnras/stad3802>`_
confirms this scope. Table 1 includes all 47 Szajna (2011) A--X (0--2)
transitions in its MARVEL input. Section 4 retains the WYLLoT (2018) X-state
potential parameters, with a refitted Born--Oppenheimer-breakdown correction;
the revised model is not an independent dissociation-energy determination.
Section 5.1 specifies X-state v<=22, J<=60 and A-state v<=1, J<=25 for AlH,
including quasi-bound levels. Continuum absorption is provided separately.
Shared spectroscopy and a constrained ground-state potential explain why
agreement with Szajna or WYLLoT cannot be counted as independent evidence.
No new state-sum convergence or mixture calculation is claimed by this
full-text update.

Controlled raw-NASA substitution
--------------------------------

The sensitivity probe replaces only the AlH formation function with the frozen
NASA ALH-AL-H reaction. It has a separate diagnostic identity and preserves
the 537-species ion-inclusive network and each pair's elemental budget.
Other element/H ratios remain fixed while C/H is changed for the three C/O
settings, before normalization.

There are 75 paired conditions: temperatures 1500, 2000, 2500, 3000 and 4000 K;
pressures 1e-6, 1e-4, 0.01, 1 and 100 bar; C/O = reference, 1 and 1.5.
Both endpoints are cold-solved at tolerances 1e-13 and 5e-14, for 300 accepted
solutions. The archive retains all species' amounts, logarithms, mole fractions,
differences, thermochemistry, formula matrix, budgets and elemental allocations.

.. list-table:: C/O = 1.5, 2500 K, 100 bar
   :header-rows: 1

   * - Quantity
     - Current curated
     - Raw-NASA AlH probe
   * - AlH mole fraction
     - 2.35246e-6
     - 2.92937e-6
   * - Fraction of Al held in AlH
     - 50.9702%
     - 63.4698%
   * - Fraction of Al held in atomic Al
     - 47.4512%
     - 35.3773%
   * - Fraction of Al held in AlOH
     - 1.14666%
     - 0.85493%

AlH increases by about 24.5% at this point and competes for aluminium with
the unchanged atomic and other molecular species. The full aluminium partition
is retained; these selected rows are not the whole budget. The largest absolute
AlH mole-fraction change is about +6.93e-7 at C/O = 1.5, 2000 K and 100 bar,
with a probe/current ratio of 1.593.

At this same 2000 K, 100 bar, C/O = 1.5 point, the largest absolute HCN change
in the grid is -1.21e-9 (about -0.0351%), while CO changes by +2.49e-7
(about +0.0443%). The prominent response is redistribution among Al carriers;
all species and conditions remain in the archived arrays.

Tighter tolerance changes x by less than 9.2e-14 and ln(x) by less than 2.4e-5.
An independent standalone FastChem comparison uses identical constants and
budgets at C/O = 1.5, 3000 K, 100 bar, with both endpoints agreeing within
4.5e-8 dex. This is a different temperature from the highlighted case.

These are ideal-gas equilibrium sensitivities, not an uncertainty interval,
an observed atmospheric response or validation of a selected source. Condensed
aluminium phases, nonideality, kinetics, transport and spectral synthesis are
outside this calculation. The elemental-share ranking depends on this scope.

Scientific assessment
----------------------

The available spectroscopy conditionally favours the NASA-like, stronger
binding energy over the old JANAF value. The reconstructed thermochemical D0
values are about 3.0502 eV for NASA and 2.9418 eV for JANAF. Converting the
1979 De value with a spectroscopic zero-point estimate of about 0.1034 eV
gives approximately 3.0566 eV at the paper's stated lower X asymptote. This
is about 0.0064 eV above NASA and 0.1148 eV above JANAF. The comparison uses
no additional atomic fine-structure subtraction. The discrepancy from NASA
is smaller than the reported 0.01 eV scale, but this is not a statistical
consistency test: the original error definition and a complete zero-point
uncertainty budget remain unavailable.

This is stronger evidence than the previously inspected, broad Barklem-Collet
uncertainty alone. It supports the direction of an energy revision, not using
3.16 eV directly as D0 or endorsing an entire NASA model from one measurement.
The 1979 access gap and the 3.17/3.16 eV discrepancy are resolved. The supplied
2011 constants also close the access gap for the displayed higher zero-point
terms: their contribution is negligible on the original 0.01 eV scale.
The Gurvich re-evaluation and Luo source details, potential-zero/reference
reconciliation, complete uncertainty treatment and high-temperature state
completeness remain open. The original article
discusses error sources but supplies no confidence-level definition. Empirical potential
asymptotes fitted mainly to low-lying levels are not independent dissociation
measurements.

Production adoption is outside this investigation change. A future correction
should justify a balanced energy reference and a thermal-function choice under
a new model identity, with the old dataset remaining reproducible. The present
raw-NASA substitution provides a numerical sensitivity result for that review.

Evidence and reproduction
--------------------------

The source and sensitivity packets are
``tests/data/validation/alh_source_review/20261102`` and
``tests/data/validation/alh_equilibrium_impact/20261102``. Their READMEs give
offline reproduction commands. Source provenance, inspection limits, selected
input/output hashes and execution environments are recorded separately from
the scientific assessment.

``mceg/contracts/alh_source_review.yaml`` separates source attribution from
fixed-network sensitivity and physical applicability. Ordinary tests require
neither network access, MCEG nor a FastChem executable::

   PYTHONPATH=src pytest tests/model_evidence/alh_source_review_test.py \
     tests/model_evidence/alh_equilibrium_impact_test.py

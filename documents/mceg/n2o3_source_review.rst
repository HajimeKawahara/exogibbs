N2O3: prioritize a consistent thermal-plus-energy candidate
===========================================================

The current ``N2O3`` row retains a JANAF-based fit. A consistent NASA thermal
function plus ATcT 1.228 energy is the next candidate to evaluate: its
pointwise formation K is 2.16--2.97 times current across 1500--4000 K.
This is a source discrepancy requiring a candidate and substitution
calculation, not an adopted replacement. Curated remains unchanged.

Identity and historical thermal assumptions
-------------------------------------------

The literal FastChem row cites Chase's 1998 JANAF compilation. The actual
`JANAF N-028 evaluation <https://janaf.nist.gov/tables/N-028.txt>`_ is dated
December 1964. The printed table and notes (Fourth Edition, nitrogen volume,
page 1626) were inspected: the ground electronic state has statistical
weight one, symmetry number one, and free internal rotation about the N--N
bond is assumed. The printed Hf(0) is explicitly unknown; its Hf(298.15)
is 82.843 +/- 0.8 kJ/mol and S(298.15)=308.54 +/- 2.1 J/mol/K.
A balanced zero-energy difference can nevertheless be derived from fixed
Hf(298.15) and thermal increments for both molecule and atoms. This is not
claiming that the printed table directly supplies a measured Hf(0).

The NASA record cites Gurvich (1989), part 1 page 338 and part 2 page
211, and has Hf(298.15)=86.630595 kJ/mol. The
`NIST CCCBDB entry <https://cccbdb.nist.gov/exp2x.asp?casno=10544737&charge=0>`_
attributes Hf(298)=86.63, Hf(0)=91.20 kJ/mol and S(298)=314.73 J/mol/K
to Gurvich. These are compilation-level checks; the full Gurvich derivation
has not been independently read.

The ATcT identity is asymmetric ONN(O)O, InChI
``InChI=1S/N2O3/c3-1-2(4)5``, consistent with the CCCBDB Cs, singlet
assignment. ONONO nitrosyl-nitrite conformers must not be silently added
as extra states of this molecule. ATcT's network contains an uncertain
relative-isomer energy, not a complete conformer partition model.

Quantified separation
---------------------

`ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
gives Hf(0) N=470.655, O=246.843 and N2O3=90.85 kJ/mol, hence balanced
D0=1590.989 kJ/mol. The JANAF-based balanced value is 1592.393 kJ/mol;
native NASA gives 1590.804204 kJ/mol. Changing energy alone has the opposite
effect to replacing both energy and thermal functions.

.. list-table:: Dimensionless formation-K diagnostics
   :header-rows: 1

   * - Temperature (K)
     - Current/native NASA
     - Same thermal + ATcT 1.228/current
     - NASA thermal + ATcT 1.228/current
   * - 1500
     - 0.470561
     - 0.893530
     - 2.158415
   * - 3000
     - 0.356260
     - 0.945267
     - 2.828838
   * - 4000
     - 0.339109
     - 0.958663
     - 2.966144

The energy-only offset uses the documented JANAF source anchor; a unique
zero-temperature atomization energy is not inferred from the finite-range
FastChem fit.

Removing the balanced energy term leaves JANAF/NASA combined molecular and
atomic thermal ratios 0.406897/0.338684/0.321254. At 1500 K the log-K
components are +0.128119 energy, -0.899723 molecular thermal and +0.000528
atomic thermal. This is predominantly a molecular thermal discrepancy;
using only a newer formation enthalpy would not resolve it. Molecular
S at 1500 K is 444.794 versus NASA 455.207 J/mol/K, while Cp is 98.764
versus 99.193 J/mol/K.

ATcT also exposes a Beattie (1957), Gurvich third-law network input for
N2O3 -> NO + NO2: delta G(300 K)=-1.84 +/- 0.07 kJ/mol. NASA gives
-2.020986 kJ/mol on the same dissociation reaction. The difference is not
within the quoted input margin. This input is not independent of ATcT,
and is a diagnostic rather than a complete validation of either hot
thermal model. The original experimental uncertainty and third-law
reevaluation should be retained in a broader physical assessment.

Footprint and concrete follow-up
--------------------------------

The archived ``curated-cos-v1`` 75-case, 537-species ion-inclusive gas grid
has maximum nitrogen inventory fraction 1.92592e-22, correctly counting two
nitrogen atoms per molecule, and maximum mole fraction 1.04124e-26. The
maximum occurs at 4000 K, 100 bar, C/O=0.588844; all cases converged. This
small hot-grid footprint does not establish irrelevance in cooler,
oxidizing atmospheres and is not a replacement solve.

Construct a continuous NASA-thermal/ATcT-1.228 candidate with matched atoms,
verify H/S/Cp and interval joins, then run a fixed-network substitution and
conservation checks. Include lower-temperature NO/NO2/N2O3 dissociation and
state/isomer scope in a broader validation rather than extrapolating the
present production model below 1500 K.

Optional underlying papers are Beattie and Bell (1957),
`10.1039/JR9570001681 <https://doi.org/10.1039/JR9570001681>`_, and Vosper
(1976), `10.1039/DT9760000135 <https://doi.org/10.1039/DT9760000135>`_. Their
DOIs and titles were checked; full texts were not obtained. Spectroscopic
source `10.1080/00268978800100601 <https://doi.org/10.1080/00268978800100601>`_
is identified through the NIST bibliography only. These are follow-ups for
stronger physical validation, not mandatory blockers for the reproducible
source review or for constructing a numerical candidate from available data.

Reproducibility and evidence boundary
-------------------------------------

All source comparisons use one bar and separate the balanced atomization
energy from molecular and atomic thermal functions. NASA polynomials retain
R=8.314510 J/mol/K; tabulated SI functions and the production reaction use
R=8.31446261815324. The zero-temperature gauge is obtained from each source's
fixed Hf(298.15) and H(298.15)-H(0), with identical treatment of all atoms.
ATcT energies use molecule and atoms from version 1.228 together. Reported
network uncertainties are not independent errors; no covariance propagation
or complete physical uncertainty is claimed.

The offline script reproduces literal source nodes and coefficients, with
no fitted replacement or new equilibrium solve. Tests independently compare
the inherited function with the named ``curated_cos`` factory, verify
thermodynamic identities, energy/thermal closure and archived stoichiometry.
MCEG supports attribution, normalized comparison and the scoped decision;
complete physical accuracy and recovery of every historical input remain
unknown. Historical recovery is a follow-up, not a prerequisite to merge
this bounded review. Production curated coefficients are unchanged.

Run ``python mceg/reviews/n2o3_source_review.py`` from the repository root.
Pinned inputs, source-access records, numerical output and a hash receipt are
in ``tests/data/validation/n2o3_source_review/20261105``.

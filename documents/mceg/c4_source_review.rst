C4 energy, state identity and thermal-source investigation
======================================================================

This 2026-11-03 investigation for #308 and #309 explains the temperature
reversal in C4 source constants. It retains the current curated fit
provisionally. Modern state-specific energy evidence favors weaker binding
than the inherited JANAF value, but a consistent high-temperature candidate
has not yet been evaluated. No coefficients or runtime behavior change.

Identity and inspected evidence
-------------------------------

The comparison is 4 C(g) -> C4(g), neutral gas at 1 bar. A common formula
does not establish equal isomer or electronic-state inventories. Linear C4
has a triplet ground state, while the rhombic isomer is a singlet. They must
not be combined by replacing the energy of one with the thermal functions
of the other, nor counted twice through both an aggregate and explicit species.

* Current curated inherits the FastChem five-coefficient fit labelled
  Chase/JANAF 1998. The `official C-143 table
  <https://janaf.nist.gov/tables/C-143.html>`_ is actually the December 1969
  evaluation converted to 1 bar. Nine selected rows were transcribed from
  the official HTML/text. All four frozen baseline H/S/Gf nodes agree.
  The `official carbon-volume PDF
  <https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-Carbon.pdf>`_ was subsequently
  inspected for the C4 evaluation notes on printed page 691 (PDF page 143).
  The underlying experimental papers remain unread.
* The `NASA 2002 report <https://ntrs.nasa.gov/citations/20020085330>`_
  was inspected using the previously downloaded official copy. All 27 C4
  coefficients match the frozen record. It cites Gurvich 1991, volume 2,
  part 1 page 18 and part 2 page 16; those pages were not obtained.
  NASA's 200-20000 K fit coverage is not a physical accuracy guarantee.
* The full author-hosted paper by `Xu et al. (1997)
  <https://doi.org/10.1063/1.474715>`_ was inspected for C4 spectra and
  assignments. It identifies linear triplet and low singlet/excited states;
  those electronic origins alone are not a complete hot partition function.
* The full `Karton et al. (2009) author manuscript
  <https://arxiv.org/abs/0812.3852>`_ was inspected for computational methods,
  state-specific atomization energies and the small linear/cyclic splitting.
  The journal DOI is `10.1080/00268970802708959
  <https://doi.org/10.1080/00268970802708959>`_.

What the original JANAF notes establish
---------------------------------------

The December 1969 model explicitly assumes linear D-infinity-h C4 with a
**singlet 1Sigma ground state**, rotational symmetry number two, equal
1.28 angstrom bonds and B0=0.171328 cm^-1. Its estimated vibrational frequencies
are 2150, 950 and 1700 cm^-1 (each nondegenerate), plus two doubly degenerate
550 cm^-1 bends. The notes describe the molecular functions as estimates.
The evaluators changed the earlier triplet assumption to a singlet and used
550/550 cm^-1 bending modes instead of Sandborn's 200/440 cm^-1 estimates
to reproduce the available vapor-pressure observations. They describe agreement
at 2000-3000 K while warning that the functions outside this interval provide
only a general trend and may not describe the real molecule.

The original energy reduction uses Drowart et al.'s second-law C4/C ion-intensity
ratio over 2200-2700 K, yielding Hf(2400 K)=226 +/- 7 **kcal/mol**.
Absolute pressures rely on the atomic-carbon vapor pressure and an estimated
ionization cross-section ratio of 4.96; the resulting S(2400 K) is
90.5 +/- 5 **cal/mol/K**. The notes separately discuss an overall entropy
uncertainty of +/- 4.5 cal/mol/K from a possible factor-two cross-section uncertainty.
These historical error statements are retained separately, without combining
them statistically. The displayed selected uncertainties are Hf(0)=964 +/- 33
and Hf(298.15)=971 +/- 33 kJ/mol, and S(298.15)=228.3 +/- 16.7 J/mol/K.
The exact numerical table anchors used by the replay remain unchanged.

Thus the inherited thermal construction has a specific obsolete state
assumption relative to the observed linear triplet; this is stronger evidence
than a compilation-date comparison. Multiplying by a triplet weight alone
would leave the estimated bends and hot-state inventory unresolved. A revised
state-consistent model is the scientific next step. Drowart et al. (1959),
Pitzer and Clementi (1959), and Sandborn (1968) were identified in the inspected
notes but their full papers were not inspected in this investigation.

Numerical cause of the reversal
-------------------------------

At 298.15 K, JANAF gives Hf=970.688 kJ/mol and S=228.322 J/mol/K;
NASA gives Hf=1033.903742 kJ/mol and S=252.861258 J/mol/K using its
historical R=8.31451 J/mol/K. The source comparison uses the frozen baseline
R=8.31446261815324 throughout and includes both sources' native C atom
references. Its balanced JANAF-minus-NASA reaction H298 offset is
-63.186188 kJ/mol and the reaction S298 offset is -24.537590 J/mol/K.

.. list-table:: Dimensionless atom-formation equilibrium constants
   :header-rows: 1

   * - T, K
     - JANAF / NASA
     - H298-aligned JANAF / NASA
     - Current curated / NASA
   * - 500
     - 194692
     - 0.0488046
     - Outside curated domain
   * - 1000
     - 88.3365
     - 0.0442279
     - Outside curated domain
   * - 1500
     - 6.85368
     - 0.0432144
     - 6.99745
   * - 2000
     - 1.87136
     - 0.0418732
     - Not sampled in frozen baseline
   * - 3000
     - 0.471947
     - 0.0374753
     - 0.465546
   * - 4000
     - 0.219449
     - 0.0328264
     - Not sampled in frozen baseline

At 1500 K the reaction Gibbs difference is -24.005342 kJ/mol: -63.186188
from H298, -0.836633 from the subsequent enthalpy increment, +36.806385
from S298 and +3.211094 from the subsequent entropy increment. At 3000 K
the sum becomes +18.729709 kJ/mol. The energy advantage of JANAF is overtaken
by its lower entropy. The atomic contributions are only +0.019342 and
+0.007684 kJ/mol at these temperatures. This is predominantly a molecular
energy/thermal-model conflict, not an atomic-reference or pressure artifact.

The inherited fit differs from JANAF by +2.10% and -1.36% at the two curated
baseline nodes. Those fit residuals cannot explain factors of about 7 and
0.47 against NASA. Aligning H298 leaves a factor 23-27 thermal difference;
an energy-only patch is insufficient. JANAF Cp is 81.955/85.822 J/mol/K
at 1500/3000 K, versus NASA 84.131858/93.607493. Thus the discrepancy is
not solely a constant entropy or spin-degeneracy factor either.
The sampled JANAF formation-G and H/S constructions agree within
0.000241 in ln K, consistent with printed rounding.

State-specific energy and spectroscopy
--------------------------------------

Balancing explicit molecular and atomic H(0) anchors gives native D0 values
1880.639 kJ/mol for JANAF and 1819.790420 kJ/mol for NASA. This uses
H(0)=Hf(298.15)+[H(0)-H(298.15)] in each source gauge, without evaluating
NASA polynomials at 0 K. The graphite-reference JANAF Hf(0)=964.101 kJ/mol
and NASA's H298-H0=13.117742 kJ/mol are different quantities and cannot
be interchanged.

Karton et al. give TAE0=429.16 kcal/mol for the linear triplet and 430.09
for the cyclic singlet: 1795.60544 and 1799.49656 kJ/mol. These include
zero-point energy and are total atomization energies, not a particular
fragmentation channel or electronic well depth. The cyclic minimum is lower
by 3.89112 kJ/mol in this calculation, while higher-order correlation terms
stabilize the linear isomer relative to lower-level calculations. Neither
the small zero-temperature energy gap nor spin multiplicity alone determines
their high-temperature abundance ratio.

These state-specific results favor weaker C4 binding than both legacy
models. NASA is closer to them, but replacing the entire model with NASA
does not follow automatically: its underlying inventory is still unverified,
and the W4 calculation does not supply a complete thermochemical model over
1500-4000 K. No uncertainty from a general method benchmark is assigned as
a C4-specific error bar here. A candidate must explicitly reconcile the
atomic fine-structure and zero-point conventions before applying an energy
shift; the native-source comparisons above do not perform that adoption step.

Xu et al. assign the first linear singlet, 1Delta_g, at 0.332 +/- 0.015 eV
above the triplet and report a low bending mode at 323 +/- 50 cm^-1.
The isolated electronic weight of that singlet relative to the triplet,
(2/3) exp[-0.332 eV/(kT)], is 0.0511 at 1500 K and 0.1846 at 3000 K.
This diagnostic excludes rovibrational differences, other states and the
cyclic isomer; it is neither a missing-state correction nor an abundance.
It must not be multiplied into a legacy partition function without first
checking which states that function already includes.

One source inconsistency is retained explicitly: section IV B and Table II
give a tentative 1Sigma_g+ origin at 0.50 +/- 0.02 eV, while Table V lists
0.93 eV. The latter entry was visually checked. This review does not choose
between them or use either in its one-state diagnostic.

Application footprint and next work
-----------------------------------

The existing 75-case, 537-species ion-inclusive current-curated archive gives
a maximum C4 carbon fraction of 1.014389e-6 (0.000101439%) and mole fraction
2.052816e-10, at 2000 K, 1e-6 bar and C/O=1.5. All original cases report
convergence. This reuses the 1500-4000 K archive; no new equilibrium solve
or one-species substitution was performed. A trace footprint in that grid
does not bound the error or establish irrelevance in carbon-rich/cooler
domains outside it.

MCEG records two scoped follow-ups:

1. Reconcile the legacy and modern state/energy inventories. Prioritize
   Gurvich 1991 volume 2 part 1 page 18 and part 2 page 16 (no verified DOI),
   and `Gingerich et al. 1994
   <https://doi.org/10.1021/ja00088a025>`_ for pressure calibration, third-law
   corrections and uncertainty. Only its publisher metadata was inspected.
   `Martin et al. 1991 <https://doi.org/10.1063/1.459747>`_ is a supporting
   source for revised partition functions; its institutional abstract was
   inspected, not the full text. The JANAF evaluation notes are now inspected;
   original experimental/calculation papers remain uninspected and can be
   checked where their reduction assumptions affect the candidate.
   Resolve the Xu singlet-entry discrepancy
   if that level is used. These are scoped evidence tasks, not an obligation
   to obtain every historical reference before merging this investigation.
2. Construct a consistent state-resolved or explicitly aggregated candidate,
   aligning atomic energies and specifying spin, symmetry, vibrations and
   isomer counting. Avoid adding states twice. Then evaluate finite C4
   substitution with fixed network/inventory, conservation, residuals and
   carbon allocation in relevant conditions. Record remaining state and
   temperature limits rather than claiming complete physical validation.

The canonical record is ``mceg/contracts/c4_source_review.yaml``. This source
investigation can be merged with those scientific follow-ups open; retaining
curated provisionally does not mean that a correction is scientifically
unnecessary. Full downloaded papers are not committed. Replay offline with::

   python mceg/reviews/c4_source_review.py
   pytest -q tests/model_evidence/c4_source_review_test.py

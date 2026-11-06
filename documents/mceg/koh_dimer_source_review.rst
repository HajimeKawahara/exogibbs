Potassium hydroxide dimer source investigation
===================================================

For neutral gas (KOH)2, **the Gurvich1997/NASA construction is the preferred
scientific improvement direction** among the inspected sources. The larger
formation constant comes from revised thermal functions, despite a less negative
formation enthalpy. This review records that reasoning and its limits; curated
coefficients remain unchanged pending a named, consistently referenced candidate
and a focused substitution check.

Identity and inspected sources
-----------------------------------

FastChem ``H2K2O2 Potassium_Hydroxide`` and NASA ``K2O2H2`` denote the neutral
six-atom gas dimer, distinct from KOH monomer, K2O2 peroxide, hydrates, ions and
condensed hydroxide. The `official JANAF H-054 table
<https://janaf.nist.gov/tables/H-054.html>`_ retains a December1970 evaluation,
although the inherited fit cites JANAF1998. All numerical comparisons below use
dimensionless equilibrium constants at 1 bar.

The inspected `JANAF1974 supplement, p.447
<https://doi.org/10.1063/1.3253143>`_ assigns C2h symmetry, rotational symmetry
number two and electronic weight one. Its vibrational modes and geometry are
estimates transferred from related molecules: six core modes from K2F2 and
O--H modes from the Li hydroxide construction. Its Hf298 is -156.5 +/-3 kcal/mol
(-654.796 kJ/mol). The printed gas association enthalpy is -45.3 +/-3 kcal/mol;
this must not be confused with total atomization energy. Historical functions
use calories and 1 atm, so the modern 1-bar table is used for replay.

`Gurvich et al. (1997), pp.1059--1067
<https://doi.org/10.1063/1.555996>`_ supports a planar D2h model with potassium
matrix spectra and gas electron diffraction, beyond analogy with sodium.
Four adopted modes are based on matrix IR, six on an ionic calculation, and
two O--H stretches are estimates. Electron diffraction helps establish structure,
but the evaluation identifies inconsistent numerical bond lengths and derived
frequencies in earlier analyses. Thus the complete molecular model is not a
direct gas-phase measurement. Its symmetry number is four and electronic weight
one; high-temperature harmonic limitations remain explicit.

The downloaded full review and historical JANAF page were inspected, including
visual checks of Tables24/25/28 and the selected energy. Original spectroscopy,
diffraction and vapor experiments are distinguished from this evaluation-level
inspection in the packet. Their complete confirmation remains unknown in MCEG.

Numerical origin of the discrepancy
----------------------------------------

The replay balances 2 K(g) + 2 O(g) + 2 H(g) -> (KOH)2(g), retaining each
source's atomic terms. The four exact JANAF nodes close the frozen #309 screen
within 6e-14 in ln K. The inherited fit contributes only +0.01374/-0.00997
in ln K relative to JANAF at 1500/3000 K.

.. list-table:: Formation constants and diagnostic reaction-H298 alignment
   :header-rows: 1

   * - T (K)
     - Curated/NASA
     - JANAF/NASA
     - H298-aligned JANAF/NASA
   * - 1500
     - 0.10103
     - 0.09965
     - 0.03295
   * - 3000
     - 0.03235
     - 0.03267
     - 0.01879

The slight difference across the factor-ten color boundary at 1500 K is a
screening artifact, not evidence that one construction is scientifically sound.
At 500 K JANAF/NASA is 3.391: the source ordering itself changes with temperature.
Aligning H298 alone makes the high-temperature discrepancy larger, demonstrating
why an energy-only correction cannot represent this update.

.. list-table:: JANAF-minus-NASA reaction Gibbs terms (kJ/mol)
   :header-rows: 1

   * - T (K)
     - Molecular H298
     - Thermal enthalpy
     - Entropy term
     - Atomic references
     - Total
   * - 1500
     - -13.800
     - -12.710
     - +55.293
     - -0.022
     - +28.761
   * - 3000
     - -13.800
     - -15.373
     - +114.570
     - -0.059
     - +85.338

Four old estimated 1250 cm^-1 bending modes become 290--300 cm^-1 modes.
The twelve-mode harmonic comparison reproduces the printed Cp differences
within 0.0021 J/mol/K at seven nodes between 298.15 and 4000 K. Combining
vibrational entropy with the rotor change,

.. math::

   \Delta S_{rot}=R[\tfrac12\ln(3.4/3.7587)-\ln(4/2)]
   =-6.1801\ {\rm J\,mol^{-1}\,K^{-1}},

reproduces the printed entropy differences within 0.011 J/mol/K. This is a
diagnostic using rounded constants, not an independently validated partition
function. NASA follows Table28 within 4.2 J/mol in H, 0.0041 J/mol/K in S and
0.013 J/mol/K in Cp at those nodes.

Using each source's monomer as well gives JANAF/NASA ratios 0.14957/0.05169
for 2 KOH(g) -> (KOH)2(g) at 1500/3000 K. This atom-cancelling comparison changes
both molecular source functions and is not the response to replacing only the
dimer. Changing standard pressure from 1 atm to 1 bar shifts an ideal-gas
atom-formation ln K by only -5 ln(1.01325), about -0.0658; inconsistent pressure
conventions cannot explain the observed high-temperature difference.

Energy ancestry and next work
----------------------------------

The 1997 evaluation adopts Hf298 = -641 +/-8 kJ/mol after reassessing
potassium vapor data. Its principal inputs are Gusarov/Gorokhov1968 and
Hastie/Zmbov/Bonnell1984. The intermediate selected reaction is
KOH(cr) + KOH(g) -> dimer(g), with H0 = +16 +/-3 kJ/mol; this is neither
gas association nor atomization. Several reconstructed total-pressure results
share that assumption and are not independent confirmations. The prose's
8 kJ/mol uncertainty and Table28's 9 kJ/mol at 0/300 K are both retained;
no Gaussian or independent-error interpretation is assigned.

The inspected publisher abstract of `Hastie et al. (1984)
<https://doi.org/10.1007/978-1-4612-5180-4_19>`_ independently reports greater
dimer importance over liquid KOH than the old JANAF evaluation. The full chapter
is uninspected subscription content. Its vaporization measurements concern a
liquid reservoir, not direct atmospheric abundance. The prior Na--K exchange
experiment is an additional shared ancestral constraint, not the sole origin of
the revised potassium evaluation.

Priority original-source confirmation is Hastie1984 (DOI above) and Gusarov and
Gorokhov, Zh. Fiz. Khim.42, 860--865 (1968), English translation Russ. J. Phys.
Chem.42,449 (DOI not verified). For structure, the verified DOI of Girichev and
Lapshina1989 is `10.1007/BF00751899 <https://doi.org/10.1007/BF00751899>`_;
only publisher-deposited bibliographic metadata was inspected. Belyaeva1978
thesis and Opt. Spektrosk.71,88 (1991) remain uninspected with no verified DOI.
The older exchange experiment is `Schoonmaker and Porter1959
<https://doi.org/10.1063/1.1730469>`_ (metadata inspected, full text uninspected).

The existing 75-case, 537-species ``curated-c2h-v1`` archive yields a maximum
dimer potassium fraction of **2.20631e-9 of the K inventory**, at 1500 K,
100 bar and reference C/O=0.5888436554; the mole fraction is 2.22699e-16.
Two K atoms per molecule are counted and archived potentials reproduce the
unchanged fit. This finite-grid footprint is not a new equilibrium calculation,
substitution response, physical error bound or evidence about cooler conditions.

MCEG retains three tasks: construct and test a named Gurvich/NASA dimer candidate,
confirm ancestral measurements, and refine soft-mode/high-temperature treatment
if the target application makes the dimer relevant. Candidate evaluation need
not wait for every old paper. Source preference, numerical reproduction and
physical/atmospheric accuracy are assessed separately. The contract is
``mceg/contracts/koh_dimer_source_review.yaml``; compact numerical excerpts and
hash receipts are in ``tests/data/validation/koh_dimer_source_review/20261103``.
Full copyrighted articles are not committed.

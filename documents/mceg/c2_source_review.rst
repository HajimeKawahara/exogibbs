C2 source, energy and thermal review
====================================

Prioritize an ATcT energy candidate while retaining the current C2 thermal
function and its fit residual. The source evidence supports a changed physical
atomization anchor; adoption still requires a separate candidate and fixed-network
substitution. This review leaves curated unchanged.

Identity and sources
--------------------

Neutral ``C2`` is the equilibrated carbon dimer with X1Sigma+g ground state;
the low a3Piu state contributes thermally. It is not a separate triplet-only
species. The current FastChem fit is JANAF-labelled. Fresh official
`JANAF C-113 <https://janaf.nist.gov/tables/C-113.html>`_ and C-003 atom tables
are pinned with selected `NASA CEA records
<https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp>`_.
The latter attributes C2 to Gurvich1991. Every coefficient matches the screening
snapshot, with original NASA interval bounds preserved.

For 2 C(g) -> C2(g) at 1 bar, the source-balanced D0 is 593.109 kJ/mol from
JANAF and 600.000010 kJ/mol from NASA. H0 comes from the printed Hf298 and
H298-H0, not a zero-temperature polynomial extrapolation. NASA dimensional
functions use its native R=8.314510; equilibrium normalization uses the modern
8.31446261815324 J/mol/K. The packet records molecular and reaction G/H/S/Cp
and separates H0, thermal enthalpy and entropy differences.

.. list-table:: Formation-constant ratios
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - Current / JANAF
     - H0-aligned JANAF / NASA
   * - 1500
     - 0.62402
     - 1.00106
     - 1.08318
   * - 2000
     - 0.70584
     - 1.00457
     - 1.06339
   * - 3000
     - 0.78974
     - 1.00186
     - 1.03910
   * - 4000
     - 0.82960
     - 0.99834
     - 1.02229

Improved energy evidence
------------------------

`ATcT version 1.228
<https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
provides Hf0(C)=711.370 and Hf0(C2)=819.940 kJ/mol, giving D0=602.800 kJ/mol.
The matching ATcT identifiers and both 0/298 K values are pinned. The general,
singlet-only and triplet-only entries have different 298 K conventions: the
zero-energy cycle uses the ground level and is not an exchange of those thermal
models. Marginal uncertainties are recorded; a reaction uncertainty is not
invented by adding correlated species errors independently.

The inspected `Borsovszky et al. (2021)
<https://doi.org/10.1073/pnas.2113315118>`_ photofragment experiment obtains
602.804(29) kJ/mol after accounting for the initial triplet level and the carbon
atomic fine-structure products. This is D0 relative to the molecular ground
level, not De. Its agreement with the versioned ATcT cycle supports the energy
change independently of the partition-function calculation.

Relative to the physical JANAF source anchor, the ATcT shift is +9.691 kJ/mol.
With the literal inherited thermal fit retained, the energy-only K multipliers
would be approximately 2.1750, 1.7910, 1.4748 and 1.3383 at the four nodes.
These are analytical diagnostics, not solved abundance responses. The future
candidate must preserve the inherited fit residual instead of equating its
1/T coefficient to the physical JANAF D0.

Modern thermal diagnostic
-------------------------

The `8states model (2018) <https://doi.org/10.1093/mnras/sty2050>`_ and
`MARVEL update (2020) <https://doi.org/10.1093/mnras/staa1954>`_ supply eight
singlet/triplet electronic manifolds. All 44189 distributed 12C2 levels, through
J=190, are archived as compact numerical columns. They already encode the
allowed identical-boson rotational states; gtot=2J+1 and nuclear spin is zero.
No extra factor of three or division by the rotational symmetry number two is
applied to this explicit state sum. The homonuclear symmetry is thus respected
without counting it twice.

The sum matches the separately distributed partition nodes within 3.4e-6. These
nodes accompany the original model, whereas the current states include the
MARVEL substitutions; exact byte-level equality is not assumed. Q/thermal-JANAF
is 0.996726--1.015605 and Q/NASA is 1.079173--1.037706. Against NASA, thermal H
changes by -852.1 to -394.8 J/mol and Cp by -0.1776 to +0.3750 J/mol/K.
The natural-mass translation convention is held fixed; the pure-12C2 isotope
translation changes S by -0.01112 J/mol/K.

The top 20 percent of distributed energy contributes at most 2.46e-5 of Q.
The small tail does not bound omitted electronic states or establish the
physical treatment of levels above dissociation. The thermal difference is much
smaller than the proposed energy effect and does not require a simultaneous
thermal replacement. A future 8states candidate should compare retained-thermal,
full-distributed and physically justified bound-level controls separately.

Importance and decision
-----------------------

The final ``curated-tio2-v1`` 75-case, 537-species ion archive gives maximum C2
mole fraction 2.873664e-6 and carbon allocation 0.767490 percent at 2500 K,
1e-4 bar, C/O=1.5. Its target function matches the named factory exactly. This
saved hot hydrogen-rich grid does not validate photochemistry or condensed
carbon regimes.

Proceed next to the retained-thermal ATcT energy candidate and substitution.
The source-review PR can stand on its pinned diagnostics; it does not require
all ancestral JANAF papers or claim candidate adoption and full physical accuracy.

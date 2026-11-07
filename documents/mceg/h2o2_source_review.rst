H2O2: retain assessed thermal functions; separate the energy update
===================================================================

Retain the current Dorofeeva-based thermal function in this source review.
A balanced ATcT 1.228 energy-only candidate is a useful next calculation.
The available APTY state sum does not support replacing the thermal model
through 4000 K. No curated data are changed here.

Current evaluation and energy
-----------------------------

The literal FastChem ``H2O2`` row cites
`Dorofeeva et al. (2003) <https://doi.org/10.1063/1.1547435>`_. The full
paper's H2O2 discussion and Table 4 (pages 894--895) were inspected, and
selected numerical nodes were independently checked against both printed
pages. The evaluation includes torsional levels, rotation/torsion
interaction and anharmonic/centrifugal corrections for ground-state gauche
HOOH. It gives Hf(298.15)=-135.880 kJ/mol, S(298.15)=234.542 J/mol/K and
Cp(298.15)=42.416 J/mol/K. Its published Cp uncertainty estimates are
0.3, 1 and 2 J/mol/K at 298.15, 1000 and 3000 K respectively.

The current five-coefficient reaction is an approximation to that source,
not a literal reproduction of all caloric nodes. At 1500/3000/4000 K,
current minus the Table-4-plus-JANAF-atoms reaction is -0.483/+0.569/+1.914
kJ/mol in H and -1.110/+1.997/-0.138 J/mol/K in Cp. These comparisons retain
the explicit source/atomic basis; the original fit objective and atom basis
were not recovered. They do not prove a molecular Cp measurement error.

`ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/index.php>`_
gives Hf(0) H=216.034, O=246.843 and HOOH=-129.400 kJ/mol. The balanced
D0 is 1055.154 kJ/mol, versus 1055.542 from the 2003 table combined with
JANAF H/O atoms. The -0.388 kJ/mol change alone decreases formation K.
The ATcT provenance identifies an important HOOH-to-2OH dissociation
threshold; that fragment dissociation is not total atomization D0.

.. list-table:: Dimensionless formation-K diagnostics
   :header-rows: 1

   * - Temperature (K)
     - Current/native NASA
     - Same thermal + ATcT 1.228/current
     - NASA thermal + ATcT 1.228/current
   * - 1500
     - 1.015941
     - 0.969369
     - 0.954707
   * - 3000
     - 0.999364
     - 0.984565
     - 0.985475
   * - 4000
     - 1.014343
     - 0.988401
     - 0.974636

The energy-only offset uses the documented table-plus-JANAF-atom anchor;
it does not claim to recover a unique zero-temperature atomization energy
from the finite-range FastChem fit. These are pointwise source/candidate
diagnostics, not replacement abundances. NASA cites Gurvich 1989/1978 and uses Hf(298)=-135.880 kJ/mol;
its thermal function is not thereby identical to the 2003 evaluation.
The legacy `JANAF H-070 <https://janaf.nist.gov/tables/H-070.txt>`_ ends at
1500 K and is kept distinct from the later full-temperature Table 4.

State-sum comparison
--------------------

The full `APTY paper <https://doi.org/10.1093/mnras/stw1295>`_ was inspected.
Its spectral completeness limit of 1250 K concerns transitions from a
restricted lower-state set; it is not a partition-function temperature
cutoff. The distributed Q table runs to 2000 K and the complete calculated
state set still has finite J=85 and 12000 cm-1 limits. The reported final-J
increment around 0.2 percent at 1500 K is not a bound on every omitted
state contribution.

APTY includes nuclear-spin weights 1 and 3 for its permitted symmetry
species. Divide its Q by the overall hydrogen nuclear-spin product 4 for
the thermochemical convention; retain rotational degeneracy and do not
apply an additional symmetry factor. Spin-normalized Q/NASA is 0.986059 at
1500 K and 0.890167 at 2000 K. Translational mass correction from natural
34.01468 u to principal-isotope 34.005479304 u gives 0.985659 and 0.889806.
This fixed-atom elemental proxy is not an isotope mixture. Missing states
and state-sum convergence must be resolved before interpreting the growing
high-temperature deficit as an improved equilibrium thermal function.
There is no extrapolation above 2000 K or derivative model built from
rounded Q values.

Footprint and follow-up
-----------------------

In the archived ``curated-cos-v1`` 75-case, 537-species gas grid, the maximum
oxygen inventory fraction is 8.12378e-9, counting both oxygen atoms, and
maximum mole fraction is 3.18177e-12. The maximum occurs at 4000 K, 100 bar,
C/O=0.588844. All archived cases converged. This is no new substitution,
and a hot equilibrium grid does not assess cooler atmosphere chemistry or
photochemical peroxide formation.

Build the energy-only candidate with the pinned same-version ATcT atoms
and molecule while retaining the current thermal function, then solve
fixed-network substitutions and check conservation. A separate thermal
update requires a smooth, complete state-sum model and comparison with the
2003 caloric evaluation. The two cited papers are already available in
full text; no mandatory literature acquisition blocks this source review.

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

Run ``python mceg/reviews/h2o2_source_review.py`` from the repository root.
Pinned inputs, source-access records, numerical output and a hash receipt are
in ``tests/data/validation/h2o2_source_review/20261105``.

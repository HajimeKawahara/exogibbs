S2O source investigation (2026-11-02)
=====================================

The S2O formation-column discrepancy in `Issue #308
<https://github.com/HajimeKawahara/exogibbs/issues/308>`_ is a documented JANAF
publication error. Retain the present curated Lodders-derived fit pending
an energy-anchor and application review. The raw JANAF formation columns
must not be used as an alternative physical model. This closes the scoped
source-diagnosis investigation (L0--L3); it does not establish high-temperature
accuracy or equilibrium abundance effects, and changes no production data.

Identity and inspected sources
------------------------------

``O1S2`` denotes neutral gas S-S-O (disulfur monoxide, CAS 20901-21-7), not
S-O-S or a cyclic isomer. The official JANAF O-011 table and the fourth-edition
notes (printed p1727, oxygen PDF p11) specify a bent molecule, point group Cs,
ground-state weight 1, symmetry number 1, and bond angle 118 degrees. The notes
use vibrational frequencies 679, 388 and 1165 cm^-1 and cite Blukis and Myers
(1965), DOI ``10.1021/j100888a011``, for their thermodynamic interpretation.
The table evaluation date is September 1965; 1998 is the compilation date.
These are unlabelled thermochemical species, not an isotope-selected network.
The notes do not establish a complete high-temperature excited-state treatment.

The NASA/TP-2002-211556 Appendix D S2O record (printed p184, PDF p192) cites
Gurvich (1989), part 1 p293 and part 2 p178, and covers 200--1000 and
1000--6000 K. All 18 coefficients in the freshly retrieved official printed
record equal those frozen by #309. The acquisition history of the earlier
full local NASA file remains unverified. The original Gurvich pages have not
been inspected, so exact state-sum equivalence remains a follow-up.

The current FastChem record explicitly cites Lodders (2004),
`DOI 10.1063/1.1611178 <https://doi.org/10.1063/1.1611178>`_. Its sections 2.2.1--2.2.2
and Tables 4--5 were inspected. The paper identifies incorrect formation H,
formation G and log K columns in JANAF editions 3 and 4, while retaining the
298.15 K enthalpy. Its historical mechanism is not established. It replaces
the old -56.484 kJ/mol anchor with -55.390 +/- 1.10 kJ/mol and uses Gurvich
thermal functions. The uncertainty's confidence level is not specified there.
Its energy evaluation combines Berkowitz, Eland and Appelman (1977) and
Norwood and Ng (1989). Their supplied full papers have now been inspected,
including the threshold cycle and its reference states described below. NASA and
Lodders consequently share thermal ancestry; database agreement is not an
independent experimental constraint.

Numerical diagnosis
-------------------

The fresh official O-011, S-006 and O-001 tables match the archived molecule
and gas-atom Hf298, H-increment, entropy and formation-G values at all four
comparison temperatures. Thus this particular discrepancy is already in the
published tables, rather than introduced by the archived local extraction.
This narrow verification does not authenticate the remaining JANAF archive.

The reaction is ``2 S(g) + O(g) -> SSO(g)``, with dimensionless Kp at 1 bar.
The following source ratios use each database's own gas atoms. Curated values
are reused from the pinned #309 execution, not a new equilibrium calculation.

.. list-table:: Disagreement measures (not uncertainty estimates)
   :header-rows: 1

   * - T (K)
     - JANAF H/S / NASA
     - Curated / NASA
     - JANAF raw minus H/S (dex)
   * - 500
     - 1.024848
     - missing
     - 1.901147
   * - 1000
     - 1.016028
     - missing
     - 3.059898
   * - 1500
     - 1.013054
     - 0.923283
     - 4.468656
   * - 3000
     - 1.010321
     - 0.940202
     - 6.173143

Missing curated values are outside its 1500--4000 K domain. At 1500 K, the
raw-column error is about 29,420 in K, entirely distinct from the 1.3% JANAF
H/S versus NASA difference. The exact two-to-one relation to SH in the frozen
1500 K residual is a diagnostic observation, not proof of a historical
arithmetic mistake or a reason to subtract an empirical correction.

Holding the JANAF atoms fixed isolates the Lodders molecular change. At
1500 K its change in G is +1.1815 kJ/mol: +1.094 from Hf298, -0.001 from the
thermal enthalpy increment, and +0.0885 from -T delta S. At the four sampled
nodes Cp agrees exactly to the printed precision; entropy differs by
-0.059 J/(mol K). This diagnostic gives K(Lodders with JANAF atoms)/K(JANAF)
=0.909614 at 1500 K. It is not the native Lodders reference convention or a
new production model. Residual differences involving the inherited fit and
its original fitting inputs have not been uniquely attributed. A green
NASA ratio cannot validate either source's physical accuracy.

Supplied primary photoionization papers
---------------------------------------

`Berkowitz, Eland and Appelman (1977)
<https://doi.org/10.1063/1.434135>`_ identify bent S-S-O and distinguish
parent-ion production from the SO+ fragment channel. Section III.B (p2185)
uses the observed fragment onset at 905.4 +/- 0.5 Angstrom and adds 0.051 eV
of mean internal energy at 300 K. This assumes that all vibrational and
three rotational degrees of freedom contribute to dissociative ionization.
The resulting reported 0 K threshold is 13.745 eV. A wavelength conversion
with current exact constants gives 13.744859 eV after the same correction,
consistent with the printed rounding; this is a reproduction of the reported
reduction, not an independent test of that thermal assumption.

The original abstract (p2183) quotes a threshold uncertainty of 0.006 eV,
whereas section III.B gives 0.01 eV. Both page images were checked. The
record preserves both values without assigning a cause or confidence level.
Subtracting the then-used SO ionization energy of 10.29 eV gives the
reported D0(S-SO) = 3.455 eV. The associated formation enthalpy is explicitly
at 0 K: -12.75 +/- 0.25 kcal/mol, using JANAF 1968 auxiliary values.
It is not an independently established modern formation-energy anchor.

`Norwood and Ng (1989)
<https://doi.org/10.1016/S0009-2614(89)87110-6>`_, section 4.1 (p148),
identify the SO adiabatic ionization energy to SO+(X2Pi3/2) as
10.294 +/- 0.004 eV. This differs from the vertical value of 10.305 eV
and the upper spin-orbit component at 10.340 eV. The 10.584 eV SSO
parent-ion threshold in section 4.2 is a different process. Their Tables
1--2 describe ionic states and do not supply a neutral SSO thermal model.

The two primary papers therefore reproduce the cycle used by Lodders:
``13.745 - 10.294 = 3.451 eV`` for ``SSO -> S + SO``. The 0.051 eV
correction is already included in 13.745 and must not be added again.
This fragment dissociation energy is distinct from atomization to 2 S + O;
the latter also needs a consistent SO/atomic reference. The original SSO
measurement is shared by subsequent compilations and ATcT, so inspecting
both papers does not create independent confirmations of each evaluation.

Modern energy constraint and remaining work
-------------------------------------------

`ATcT 1.172 species 631
<https://atct.anl.gov/Thermochemical%20Data/version%201.172/species/?species_number=631>`_
identifies the same SSO connectivity and CAS number. It gives Hf0=-50.62 and
Hf298=-52.63 +/- 0.43 kJ/mol (estimated 95% confidence limits). Using its
neutral S and O values gives a balanced atomization energy
``2*276.83 + 246.844 - (-50.62) = 851.124 kJ/mol`` from rounded published
values. This is +1.124 kJ/mol relative to Lodders' 850.00, not the molecular
formation-enthalpy difference alone. ATcT reports a 58.8% SSO--S correlation;
independent-error quadrature would be inappropriate. Berkowitz (1977) supplies
22.7% of SSO's provenance, alongside later Karton calculations. The network
is therefore partly shared with the older evaluations. It provides an energy
candidate, not replacement heat capacities or entropy.

The publisher abstract of Grant et al. (2009),
`DOI 10.1021/jp905847e <https://doi.org/10.1021/jp905847e>`_, identifies a
CCSD(T)/CBS study with core-valence, relativistic, spin-orbit and zero-point
corrections. Its SSO numerical tables were not inspected and are not adopted.
The Berkowitz and Norwood full-text access gaps are now closed. Next steps
are to inspect the Gurvich thermal inventory and remaining theoretical
records, reconcile balanced energy anchors and correlated uncertainties,
then test a scoped SSO candidate on the fixed ion-inclusive network with
conservation and mass-action residuals. No atmosphere, spectrum, condensate
or mixture-wide physical validation follows from this source review.

Offline evidence and MCEG
-------------------------

The :download:`source record
<../../tests/data/validation/s2o_source_investigation/20261102/sources.json>`
contains acquisition hashes, locations, identity, quoted numerical inputs,
search queries and access limits. Full downloaded publications remain in
ignored ``results/``; small JANAF tables and the NASA coefficient extract are
archived. SHA256 values of unarchived publications identify inspected bytes,
not an assertion that those bytes are redistributable or available offline.
The :download:`comparison
<../../tests/data/validation/s2o_source_investigation/20261102/comparison.json>`
and :download:`receipt
<../../tests/data/validation/s2o_source_investigation/20261102/receipt.json>`
replay without network access or an MCEG installation:

.. code-block:: bash

   python mceg/reviews/s2o_source_investigation.py --check
   python -m pytest -q tests/model_evidence/s2o_source_investigation_test.py

The :download:`MCEG contract
<../../mceg/contracts/s2o_source_investigation.yaml>` separates supported
source diagnosis and numerical reproduction from unknown physical accuracy
and unrun abundance effects. Optional validation uses:

.. code-block:: bash

   python -m mceg check mceg/contracts/s2o_source_investigation.yaml \
     --assessment numerical_replay_review --root .

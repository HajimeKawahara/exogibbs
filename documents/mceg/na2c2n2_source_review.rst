Na2C2N2 neutral dimer identity and source review
================================================

Resolve the identity as neutral (NaCN)2 gas and retain its current function.
The six-atom dimer is not the three-atom monomer, a four-atom hypervalent
Na2CN molecule, a cyanogen complex without metals, or condensed cyanide.
Source inspection identifies concrete thermal/energy follow-ups rather than
an immediately justified coefficient replacement. No new equilibrium solve
or curated change is made.

Printed identity and model
--------------------------

Official JANAF C-135 and printed page 683 explicitly identify
``C2N2Na2`` as ideal gas. The March 1966 evaluation assumes a
planar D2h structure, sigma=4, ground weight 1, and 12 vibrational degrees.
The source uses estimated geometry and frequencies; a crystal-based analogy
is not an experimental gas dimer structure. Hf298=-8.786
+/-13 kJ/mol is kept distinct from dimer dissociation and
from total atomization into six neutral atoms.

The complete official NASA input has no matching neutral dimer record.
NASA comparisons remain null; the frozen 93-row numerical screen is not
expanded with a monomer or a hypervalent Na2CN surrogate. The appendix identity
is resolved without inventing a NASA counterpart. The printed 12-mode model
reproduces JANAF H-H298 within 5 J/mol, S within 0.005 J/mol/K and Cp within
0.003 J/mol/K at the reviewed nodes. This confirms the inherited statistical
model, not measured modern dimer thermal accuracy.

.. list-table:: Existing source comparison
   :header-rows: 1

   * - T (K)
     - Current / JANAF
     - Source dissociation H1000 (kJ/mol)
   * - 1500
     - 1.01300200
     - 191.646
   * - 2000
     - 1.00216245
     - 191.646
   * - 3000
     - 0.99667586
     - 191.646
   * - 4000
     - 1.00300669
     - 191.646

The packet includes source molecular and atom-balanced formation G/H/S/Cp,
physical total D0, dimer-to-monomers D0/H298/H1000, analytic retained-fit
calorics and source closure. SI R is used for JANAF; any NASA comparison
retains its native R=8.314510. Each monomer reference is copied into this
packet, so the review is independent of the separate monomer-review PR.

Primary energy evidence and actionable limit
--------------------------------------------

The publisher-registered primary abstract of `Porter (1961)
<https://doi.org/10.1063/1.1731908>`_ establishes monomer/dimer vapor near
1000 K from single/double-oven Knudsen-cell mass spectra. It reports
Delta-H1000[(NaCN)2 -> 2 NaCN]=41.0 +/-3.0 kcal/mol, or
171.544 +/-12.552 kJ/mol. Its complete body was not obtained. The selected
JANAF monomer and dimer give 191.646 kJ/mol at 1000 K, 20.102 kJ/mol higher
than that central value. This exceeds the experiment's quoted uncertainty
alone; it is not a combined significance test because source correlation and
both adopted formation-energy uncertainties are unresolved.

The current dimer dissociation D0 is 196.246 kJ/mol and H298 is
197.318 kJ/mol. None of these is the molecule's total atomization D0. The old
JANAF dimer fit also uses vapor-pressure and estimated-thermal information;
its selected monomer is not silently replaced to force agreement with Porter.
A follow-up should jointly reinterpret the monomer and dimer energy/thermal
ledger, then seek gas dimer structural or computational inputs. Modern
T-shaped monomer measurements alone cannot determine the dimer's 12 modes.

Importance and reproducibility
------------------------------

Across 75 archived curated-tio2-v1 endpoints, the maximum Na share is
6.77955758e-09% at 1500 K,
100 bar, C/O=1.5; maximum mole fraction is
9.67313998e-17. All endpoints converge with 537 species.
This hot H-rich trace abundance does not validate cold salt-rich vapor or
condensation. Missing full source bodies are focused follow-ups, not a demand
for every historical ancestor before merging the documented source review.

Run ``python mceg/reviews/na2c2n2_source_review.py`` and
``pytest tests/model_evidence/na2c2n2_source_review_test.py``. The packet is
``tests/data/validation/na2c2n2_source_review/20261106`` and contract is
``mceg/contracts/na2c2n2_source_review.yaml``. Inputs use historical hash
resolution; outputs are directly pinned. Complete physical accuracy remains
unknown, distinct from the supported identity/source/retention assessment.

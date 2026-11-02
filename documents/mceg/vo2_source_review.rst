VO2 source investigation
========================

This review follows #308/#309 for neutral gaseous vanadium dioxide, ``O2V1``.
The evidence favours reassessing the old JANAF energy toward weaker binding,
with a separate reconstruction of the thermal functions. Neither raw NASA
nor a complete new replacement is adopted. The current coefficients remain
operationally available while the unresolved inputs are reconciled; this is
not a finding that their physical accuracy has been established.

Why VO2 is a priority
---------------------

In the existing 75-condition, 537-species ``curated-c2h-v1`` archive, VO2 holds
at most **62.7841% of the vanadium**, at 1500 K, 100 bar and reference
C/O = 0.5888436553555908. Its mole fraction there is only
8.56902e-9: a small total abundance can still control a trace element's
allocation. This is a finite **gas-only** grid, with condensation omitted,
not a statement about the dominant observable species in a real atmosphere.
The audit replays all 75 saved cases and records all V-bearing species at
that point. No new source-substitution equilibrium calculation is claimed.

Identity and source lineage
---------------------------

The target is the bent, neutral dioxo molecule O--V--O, with a
:math:`\widetilde X\,^2A_1` ground state and :math:`C_{2v}` symmetry.
Solid VO2, the condensed V2O4 formula used in some compilations, VO2 ions,
and peroxo/superoxo structures are distinct. The legacy table uses molar
mass 82.9403 g/mol; no isotope-specific partition function is adopted.

The current FastChem-derived record cites JANAF. Fresh official
`JANAF O-076 <https://janaf.nist.gov/tables/O-076.html>`_ and the V/O atom
tables reproduce the frozen baseline. Across 26 exact nodes from 1500 to
4000 K, the current fit differs from the reconstructed JANAF reaction
ln K by at most 0.006291, approximately 0.63% in K.
The inspected evaluation is dated December 1973, converted to 1 bar;
``1998`` names the compilation, not a new measurement.

The official pinned NASA CEA VO2 record cites **Gurvich 1982, part 1 p64,
part 2 p66**. All selected V/O/VO2 coefficients match the local frozen
records, but the official V and VO2 lower limits are **300 K**, whereas the local
copy has 200 K. The original Gurvich volume remains uninspected.
The NASA header gives formation enthalpy -232.697655 kJ/mol and
H(298.15)-H(0) = 10.622449 kJ/mol.
The 298.15 K polynomial values below are a reconstruction of the frozen
local extension, just outside the official V/VO2 intervals; they do not
establish an official low-temperature extrapolation policy.

Balanced numerical comparison
-----------------------------

All K values are dimensionless at 1 bar for
:math:`\mathrm{V(g)+2O(g)\rightarrow VO_2(g)}`. Differences mean JANAF minus
NASA, including both atom references. A common
R = 8.31446261815324 J/mol/K is used in the frozen comparison.

.. list-table:: Source ratios and energy alignment
   :header-rows: 1

   * - Temperature (K)
     - Current / NASA
     - JANAF / NASA
     - JANAF / NASA after reaction H298 alignment
   * - 1500
     - 1.49240
     - 1.48304
     - 1.72192
   * - 3000
     - 1.65731
     - 1.66098
     - 1.78976
   * - 4000
     - 1.67665
     - 1.67428
     - 1.77072

The reconstructed reaction H298 difference is +1.86261 kJ/mol.
Although the molecular formation-enthalpy headers nearly agree, the atomic
V reference contributes about +1.794 kJ/mol to the reaction G difference
at 1500 K. Comparing only the molecular headers would miss that effect.
Equalizing the balanced reaction energy leaves a larger K disagreement:
the thermal difference matters and a newer compilation date cannot settle it.
The audit preserves the H298 anchor, enthalpy increment, reference entropy,
entropy increment and atomic contributions separately. Energy alignment is
a diagnostic, not a physical correction or an abundance ratio.

What the energy evidence supports
---------------------------------

.. list-table:: Reported energies, with their original roles
   :header-rows: 1

   * - Source
     - Formation enthalpy near 298 K (kJ/mol)
     - Total atomization D0 (kJ/mol)
     - Inspection
   * - JANAF
     - -232.630 +/- 41.8
     - 1236.342, reconstructed with JANAF atoms
     - Original evaluation and tables
   * - Balducci, Gigli and Guido (1983)
     - -174 +/- 14
     - 1177 +/- 18
     - Publisher-deposited primary abstract
   * - Gunina, Lopatin and Shugurov (2012)
     - -185 +/- 15
     - Not reconstructed
     - Primary abstract
   * - Osina and Gorokhov, MSU online evaluation
     - -160.259 +/- 12.3
     - 1166 +/- 12
     - Full expert reevaluation

The displayed uncertainties are the sources' stated limits. They are not
converted into a shared confidence level, statistically averaged, or
propagated as independent errors. D0 here is total atomization to V + 2 O,
not a single VO--O bond energy or electronic well depth De.

The `1983 all-gas mass-spectrometric work
<https://doi.org/10.1063/1.445682>`_ is the strongest identified later
experimental reason to question the stronger JANAF binding. The
`MSU evaluation <https://www.chem.msu.ru/rus/tsiv/V/print-VO2.html>`_
reanalyzes that work, preferring isomolecular pressure ratios to absolute
pressures and using updated auxiliary data. The
`2012 paper <https://doi.org/10.1021/ic201644f>`_ reports a critical
reevaluation of previous VO/VO2 thermochemistry. These are related
assessments, not three independent measurements. Full 1983/2012 reaction
data, thermal corrections, V/VO/EuO references and covariances are still
needed to choose one consistent value.

The `2024 cluster study <https://doi.org/10.1051/0004-6361/202347693>`_
also identifies this discrepancy. Its inspected arXiv manuscript gives
VO2 electronic binding 1154.7 kJ/mol in Table 2 but 1150 in Table 3;
its precise De/D0/ZPE and sign conventions need reconciliation before
numerical reuse. The assumed JANAF uncertainty of 4 kJ/mol in Table 2 is
**not** the 41.8 kJ/mol uncertainty printed on the VO2 evaluation.
Its B3LYP/cc-pVTZ choice is benchmarked using VO, and its mixture comparison
changes VO and VO2 together at 1 bar with Asplund 2009 abundances.
Those results cannot be treated as an independent validation of every VO2
thermal state or a VO2-only response in this network.

Thermal states: inspect the primary page
----------------------------------------

JANAF adopts a constant ground electronic weight of 2, estimated frequencies
970, 300 and 930 cm^-1 and a bent geometry. The ground-only nonlinear
rigid-rotor/harmonic-oscillator heat capacity reproduces six checked table nodes
within 0.001 J/mol/K. This agrees with the MSU description of omitted
electronic excitations.

A reading trap on **printed p1763, Oxygen PDF p47** matters: the paragraph
about 620 and approximately 14800 cm^-1 first describes the **atomic V++
ion**. It then says a hypothetical VO2 state at 620 cm^-1 with weight 6
*would* raise S298 by 1.06 cal/mol/K, within the estimated uncertainty.
It does not adopt that hypothetical molecular state. The archived original
page and the heat-capacity closure prevent that analogy being mistaken for
an existing molecular level model.

`Kim, Weichman and Neumark (2014)
<https://doi.org/10.1063/1.4861667>`_ resolve the first three neutral states
using cryogenically cooled anions. Their assigned doublet states lie at
0, 0.1845(8) and 0.8130(5) eV; measured ground-state symmetric stretch and
bend fundamentals are 960(3) and 291(2) cm^-1. The antisymmetric stretch is
not observed in that experiment. Their electron affinity 1.8357(5) eV
revises an earlier 2.03 eV assignment; electron affinity alone cannot supply
neutral atomization energy.

`Hendrickx and Tran <https://doi.org/10.1021/ct500526m>`_ proposed alternative
assignments involving low-lying anion states. The
`authors' reply <https://doi.org/10.1021/ct5007949>`_ defends the cold-ion
assignment using orbital transitions, vibrational patterns and preparation
conditions. This dispute and the reply's shared experimental ancestry are
recorded. The resolved spectra provide a stronger basis for rebuilding the
level model, but do not constitute a complete high-temperature partition sum.

As an explicitly limited diagnostic, replacing just the electronic level
list while keeping every state's rovibrational factor identical gives
Q_el(2014)/2 = **1.2418, 1.5329, 1.6801** at 1500, 3000 and 4000 K.
This temperature-dependent electronic correction is not a complete
replacement K. State-specific rotation and vibration, anharmonicity, higher
states and isotope conventions still need treatment. The saved electronic
enthalpy, entropy and heat capacity moments make that limitation testable.

Reproduction and MCEG
---------------------

``tests/data/validation/vo2_source_review/20261102/`` contains official
tables, selected NASA records, the two relevant JANAF pages, factual
literature inputs, access limitations and source hashes. Research PDFs
listed as unarchived are not downloaded by tests. Baseline-001 and the
production coefficients are unchanged.

.. code-block:: bash

   python mceg/reviews/vo2_source_review.py
   pytest -q tests/model_evidence/vo2_source_review_test.py
   python -m mceg check mceg/contracts/vo2_source_review.yaml \
     --assessment source_attribution_review --root . --plain

The contract separates supported numerical source attribution and a
conditional preference for weaker binding from **unknown** complete thermal
accuracy and atmospheric accuracy. The next adoption decision requires a
consistent energy network and a complete thermal model, followed by a
VO2-only finite substitution and physical tests appropriate to the intended
atmosphere, including relevant condensed phases.

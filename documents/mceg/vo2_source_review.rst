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
     - 1177 +/- 11 in the selected full-text result; +/- 18 in the abstract
     - Full article, including scanned tables and error discussion
   * - Gunina, Lopatin and Shugurov (2012)
     - -185 +/- 15
     - Not reconstructed
     - Primary abstract and supporting information; main article pending
   * - Osina and Gorokhov, MSU online evaluation
     - -160.259 +/- 12.3
     - 1166 +/- 12
     - Full expert reevaluation

The displayed uncertainties retain each source's own definition. They are not
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
assessments, not three independent measurements. The full 1983 article and
2012 supporting information are now inspected. The 2012 main-text
reevaluation, a consistent modern set of auxiliary data and thermal
corrections, and shared-reference covariances remain unresolved.

What the supplied full text resolves
------------------------------------

Balducci's Table I and the discussion on printed pp5617-5618 identify the
energy chain. With all species gaseous, the selected reaction enthalpies
at 0 K are:

.. list-table:: The 1983 experimental paths
   :header-rows: 1

   * - Reaction
     - Selected H0 (kJ/mol)
     - Measured temperatures (K)
   * - VO + Eu -> EuO + V
     - 158.5 +/- 6
     - 1913-2178
   * - VO2 + V -> 2 VO
     - -74 +/- 6
     - 1913-2186
   * - VO2 + Eu -> EuO + VO
     - 84.5 +/- 6
     - 1913-2282

Their common auxiliary value D0(EuO) = 467 +/- 6 kJ/mol first gives
D0(VO) = 625.5 +/- 8.5 kJ/mol. Both subsequent paths then give
D0(VO2) = 1177 kJ/mol. The offline audit reproduces that arithmetic and
the reaction-cycle closure. The experiment uses Knudsen-effusion mass
spectrometry, with fragmentation corrections and estimated molecular
ionization cross sections; it is not a direct observation of a dissociation
threshold.

The **uncertainty differs within the paper**: Table I gives +/-18 and
+/-12 kJ/mol for the two paths, while the selected average on p5618 has
+/-11 kJ/mol. The abstract instead prints +/-18. The authors describe
selected-reaction and auxiliary errors as estimated standard deviations,
assume normally distributed effects and describe three times the quoted
errors as reliable limits. We preserve these distinct statements and do
not resolve the abstract/body difference by silently choosing one error.

Although the authors call the reaction paths independent, their inferred
energies share D0(VO) and D0(EuO). At fixed reaction heats, increasing the
EuO anchor by 1 kJ/mol raises **both** inferred VO2 energies by 2 kJ/mol.
This checked sensitivity prevents the paths being counted as independent
absolute-energy determinations. No covariance matrix is supplied, and no
new combined uncertainty is claimed.

The Appendix and Tables IV-V also expose the thermal assumptions: ground
electronic weight 2, frequencies **983, 538, 1034 cm^-1**, a 95-degree bond
angle and estimated 0.163 nm bonds. V and VO thermal functions are taken
from JANAF; the Eu/EuO functions and energy anchor trace to earlier work.
The 538 cm^-1 bend differs strongly from the later resolved 291(2) cm^-1
ground-state bend. Table IV gives H(1500)-H(298.15) = **63.590 kJ/mol**,
1.088 kJ/mol below JANAF. These historical assumptions must be reconsidered
when reducing the measured equilibria with a modern thermal model.
The audit compares enthalpy increments only: Table IV does not explicitly
state standard pressure, so its reconstructed entropy is kept in its
printed convention without imposing a pressure-normalized Q ratio.

The supplied **Gunina supporting information does not contain the main-text
VO/VO2 reevaluation**. Tables S1-S3 contain partial pressures and reaction
heats for mixed W/Mo/V oxides; the remaining sections describe
VMoO4/VWO3/VWO4 structures. Captions label reactions 10-14 while column
labels and the cover use 7-11. The missing article is needed to resolve
that numbering and the auxiliary references and error definition behind
Hf298(VO2) = -185 +/- 15 kJ/mol. This supporting information alone does
not establish an independent VO2 atomization result.

The directional preference for weaker binding is strengthened by this
inspection, but a replacement still needs the energy and thermal model to
be evaluated together. Neither the 1983 ground-only model nor an isolated
1177 kJ/mol substitution is adopted.

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
listed as unarchived, including the user-provided Balducci article and
Gunina supplement, remain local and are not downloaded by tests. Their
basenames and SHA-256 hashes identify the inspected copies; only factual
inputs and original review notes are committed. Baseline-001 and the
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

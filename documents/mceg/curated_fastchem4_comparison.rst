Curated gas v2 compared with updated FastChem 4
==============================================

The comparison for Issue #308 uses ``curated-gas-v2`` and the full standard
gas table in `FastChem 4.0.4
<https://github.com/NewStrangeWorlds/FastChem/releases/tag/v4.0.4>`_, released
on 2026-11-03. The table is pinned to commit
``0c4b448166e1d0a0f01e3416bf7da0082ef0546e``. Its preceding
`HCN and atomic-ion update
<https://github.com/NewStrangeWorlds/FastChem/commit/15f5d7030d13771eb6bc9c8b74f2e81b35c75234>`_
changes HCN and 47 atomic-ion records relative to v4.0.3. Only neutral
molecular formation constants are compared here; ion changes would also
matter in a future mixture-abundance calculation.

File selection and convention
-----------------------------

The selected ``input/logK/logK.dat`` is byte-identical to the previously
received HCN-update table. Its HCN row now cites Goos/Burcat 2023. Both
official v4.0.4 ``wo_ions`` files still contain the old JANAF HCN row.
Therefore this study takes the neutral molecular records from the updated
full table. The standard and extended tables have identical coefficients
and compositions for all 108 selected records.

For atoms forming one neutral molecule, at standard pressure 1 bar,

.. math::

   K_p = \frac{p_{\rm molecule}/p^\circ}
               {\prod_j (p_{{\rm atom},j}/p^\circ)^{\nu_j}},
   \qquad
   \frac{K_{p,\rm curated}}{K_{p,\rm FC4}}
   = \exp[-h_{\rm curated}(T)-L_{\rm FC4}(T)].

The published fit is the natural logarithm
``L = a1/T + a2*ln(T) + a3 + a4*T + a5*T**2``.
The inspected FastChem C++ applies a separate ``k_B*T/p_standard`` conversion
when solving number densities. That conversion is not part of this
dimensionless comparison. No additional gas-constant rescaling, atomic
subtraction or zero-point correction is applied. The native gas constant
used to construct every original fit is not specified by these files.

The numerator is the named ``curated_v2`` factory, which was the public
default at execution. The repository's general ``fastchem4`` preset still
ships its pinned v4.0.3 inputs; this review explicitly supplies the archived
v4.0.4 full table and changes no runtime preset or curated coefficient.

Results and interpretation
--------------------------

All 108 tracked molecules are evaluated at 1500, 2000, 2500, 3000 and
4000 K: 540 comparisons, without equilibrium abundance solves. At 1500 K,
the original 93-row screen has 87 green, 4 yellow and 2 red comparisons
against FastChem 4.0.4. Across all 108 records these counts are 95, 7 and 6.
The symmetric difference thresholds remain factors 2 and 10.

.. list-table:: HCN formation-constant ratios, with curated as numerator
   :header-rows: 1

   * - Temperature (K)
     - Curated / updated FC4 4.0.4
     - Curated / old FC4 4.0.3
   * - 1500
     - 1.03356972
     - 1.57408785
   * - 2000
     - 1.03398310
     - 1.41537837
   * - 2500
     - 1.03373789
     - 1.32801123
   * - 3000
     - 1.02938623
     - 1.26819847
   * - 4000
     - 1.01412178
     - 1.18632833

At 1500 K, curated HCN Kp is 3.35697% above the updated FastChem value;
the reciprocal ratio is 0.96752061. This is a difference between formation
constants, not an HCN abundance change or a calibrated error estimate.

The large remaining differences include C2O (0.000529466), C2H
(0.00189863), VN (5.96337), and, outside the original NASA screen, CNC
(0.0000653875), HCP (0.00676527) and P4S3 (0.0107417). The separate
per-molecule investigations explain the source choices. Agreement can also
reflect shared ancestry: VO2 is exactly 1 in this comparison while its
thermochemical reassessment remains unresolved. NH3 is 0.93205549. Merging
their source investigations does not adopt a new VO2 or NH3 model.

Current curated/NASA values are recomputed for the original 93 mappings,
using the same frozen NASA denominator. The 14 historical NASA exclusions
and the additional C3H record now have FastChem comparisons, but receive
no newly inferred NASA mapping. The original Issue body and original
JANAF/NASA baseline remain archived. No screening grade establishes
source superiority or physical abundance accuracy.

Reproduction and evidence
-------------------------

The packet is in
``tests/data/validation/curated_fastchem4_comparison/20261106``.
``comparison.csv`` contains all five temperatures; the table below projects
1500 K. Source hashes and actual execution details are in ``inputs.json``,
``sources.json`` and ``receipt.json``. Numerical replay is offline::

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=src \
     python mceg/reviews/curated_fastchem4_comparison.py

The native FastChem provider and an independent scalar evaluation agree
within 2.85e-14 in ln(K). An independent 70-digit Decimal audit verifies
all 540 FastChem values and 465 frozen NASA values, with maximum ln(K)
differences of 5.03e-14 and 1.49e-13 respectively. These are numerical
consistency checks; no FastChem mixture solve is represented by them.
MCEG records the supported source selection and sampled replay, while
complete physical accuracy remains unknown.

.. csv-table:: Formation-constant ratios at 1500 K
   :file: ../../tests/data/validation/curated_fastchem4_comparison/20261106/table1500.csv
   :header-rows: 1

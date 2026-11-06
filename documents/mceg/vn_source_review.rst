VN neutral dissociation-energy source review
=============================================

Retain production for this source-only PR and prioritize a direct-D0 energy
candidate. Modern neutral-VN threshold spectroscopy supports an approximately
22.27 kJ/mol increase over the old JANAF physical dissociation energy. The
present NASA discrepancy is much smaller and is not a measure of that missing
modern energy information. No curated coefficient or equilibrium solve changes.

Identity and inspected evidence
--------------------------------

The canonical species ``N1V1`` is neutral diatomic VN gas, ground X 3Delta,
forming from neutral ground-state V and N. It is distinct from solid vanadium
nitride and VN+. JANAF N-018 and the NASA9 record (Chase 1998, p.1616) share
source ancestry; agreement is not independent physical confirmation. The
molecule and both atoms' official tables and literal NASA coefficients are
pinned, including the separate atomic energy gauges.

The author-institution abstract of Johnson, Davis and Morse (2016),
`DOI 10.1063/1.4953782 <https://doi.org/10.1063/1.4953782>`_, explicitly reports
neutral ground-atom D0=4.9968(20) eV, or 482.117908
kJ/mol. Its cation value 4.6871(20) eV is not interchangeable. The original full
threshold analysis was not obtained; this access limitation does not prevent
a bounded energy-candidate recommendation using the reported neutral result.

The complete 2023 author manuscript,
`DOI 10.1080/00268976.2023.2248292 <https://doi.org/10.1080/00268976.2023.2248292>`_,
section 3.3 and Table 5, gives an independent s-ccCA D0 of 481.89 kJ/mol,
0.228 kJ/mol below the measurement.
This agreement supports the central energy selection; it is not a new measured
uncertainty or validation of complete VN thermal functions.

The primary indexed abstract of the 2026 ion-cycle study,
`DOI 10.1063/5.0349800 <https://doi.org/10.1063/5.0349800>`_, reports
D0(VN+)=4.688(12) eV and cycle closure within 12 meV. It supports the consistency
of the neutral/cation cycle, but its cation number is not a replacement VN D0.
The complete 2026 uncertainty budget was not inspected.

The full 2025 Kratzer-model article,
`DOI 10.1038/s41598-025-17514-z <https://doi.org/10.1038/s41598-025-17514-z>`_,
was also inspected. Its input De=48000 cm-1 is a potential-well depth, not the
measured D0. Its 101325 Pa convention would require an entropy increase of
0.109443159 J/mol/K at 1 bar. The NIST/JANAF
comparison uses an evaluated source, not a new calorimetric measurement. This
single-potential calculation does not establish a complete excited-electronic
thermal replacement, and none of its thermodynamics is adopted here.

Native sources and bounded energy diagnostic
--------------------------------------------

All reviewed formation functions use 1 bar. Source-native NASA R=8.314510 is
kept distinct from SI R=8.31446261815324 J/mol/K. Balanced physical source D0
is 459.848000 kJ/mol for JANAF and
461.642316 kJ/mol for NASA. After D0
alignment, the two source thermal functions agree to about 0.04% at the four
nodes. The packet records molecular and reaction G/H/S/Cp, raw JANAF closure,
energy/molecule/atom decomposition and literal current-fit derivatives.

.. list-table:: Formation K diagnostics, not finite abundance responses
   :header-rows: 1

   * - T (K)
     - Current / NASA
     - D0-aligned JANAF / NASA
     - Energy-only current K factor
   * - 1500
     - 0.86632423
     - 0.99990967
     - 5.963372
   * - 2000
     - 0.89779534
     - 0.99968671
     - 3.816093
   * - 3000
     - 0.93101258
     - 0.99973771
     - 2.442002
   * - 4000
     - 0.94760080
     - 1.00003790
     - 1.953482

The last column is exp((D0(measured)-D0(JANAF))/(RT)), retaining the current
thermal shape and fitted-proxy residual. It is a proposed source sensitivity,
not a new model implementation or network calculation. Its G and H shift are
-22.269908 kJ/mol and its S/Cp shifts are zero.
The reported 0.0020 eV uncertainty is 0.193 kJ/mol; this does not cover inherited
thermal or atomic-reference uncertainty. Selecting a modern Hf0 without matching
atomic energies would obscure the direct measured bond-energy information.

Current importance, decision and replay
---------------------------------------

The final ``curated-tio2-v1`` archive contains 75 converged ion-inclusive,
537-species endpoints. VN's maximum V share is 0.01106761%
at 3000 K, 100 bar and C/O=0.588844.
This is a finite hot H-rich inventory, not a limit for all atmospheres. The
large proposed K factor warrants a separate fixed-network energy-candidate
calculation even though the baseline share is small. That calculation should
preserve old thermal functions and fit residuals, and test the neutral direct
D0 before considering a separate electronic/rovibrational update.

The source review is mergeable under the established bounded-review standard.
Complete physical uncertainty, missing-state effects and atmospheric abundance
accuracy remain unknown; unavailable historical full texts are not an unlimited
merge gate. Run ``python mceg/reviews/vn_source_review.py`` and
``pytest tests/model_evidence/vn_source_review_test.py``. The pinned packet is
``tests/data/validation/vn_source_review/20261106`` and MCEG contract is
``mceg/contracts/vn_source_review.yaml``. Input hashes use the historical resolver;
comparison output hashes are direct. No complete paper is committed.

HNO3: source review on the current curated baseline
===================================================

Retain the present Dorofeeva 2003-based curated fit. The actual adopted source has been inspected and reproduced; no scientifically compelling full replacement is established by comparison with the older JANAF web table or finite-range opacity partition data.

Identity and inspected sources
------------------------------

Neutral molecular nitric acid HONO2, CAS 7697-37-2, planar singlet X1A-prime gas species with internal OH torsion; not nitrate ion, aqueous acid, liquid acid or HONO.

The current source label is **Dorofeeva 2003 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1989 pt1 p367 pt2 p231. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

The current fit cites Dorofeeva 2003, not the older JANAF 1998 web table. Table 1 of the actual paper gives a molecular torsional evaluation that the inherited fit reproduces within 1.84% at the four hot nodes when balanced against JANAF atoms. The approximately 20% difference from the old web table is therefore an intended source distinction. ATcT changes this actual-source atomization energy by only -0.200 kJ/mol, within the paper formation-enthalpy uncertainty 0.5 kJ/mol. This is a source-based reason to retain, not merely a trace-abundance argument.

Dorofeeva et al. 2003 was inspected in full and four hot table nodes plus its 0/298 K anchors were transcribed. The work treats internal torsion, omits electronic states above 30000 cm-1 and states caloric uncertainty estimates. TIPS 2025 principal-isotope Q/NASA is 0.9214 at 1500 K and 0.9207 at 3000 K, but its table ends at 3500 K and is not a demonstrated complete thermal improvement. ExoMol 2015 advertises line-list use only up to 500 K, so it cannot be promoted into a 1500-4000 K replacement from the abstract.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

The accepted molecular source includes internal rotation; no extra conformer multiplier is imposed. TIPS 14N/1H nuclear-spin product 6 is removed once, with molecular symmetry already included. Use the same natural-mass translation for comparison. The true adopted-source energy update is distinct from the explicitly labelled JANAF-proxy energy diagnostic.

The balanced source difference is split into energy, molecular thermal and atomic thermal terms before exponentiation. Energy alignment removes the source-specific energy-over-R difference while retaining molecular and atomic thermal differences. This is a diagnostic, not a replacement. The current fit has an inherited residual; its 1/T coefficient is not a physical dissociation energy. All source-specific molecular and reaction G/H/S/Cp and analytic current-fit calorics are in ``comparison.json``.

.. list-table:: Exact source knots; K ratios are equilibrium-constant diagnostics
   :header-rows: 1

   * - T / K
     - Current / NASA
     - JANAF / NASA
     - Energy-aligned JANAF / NASA
     - ATcT energy-only / current
     - NASA thermal + ATcT / current
   * - 1500
     - 1.049061
     - 0.86888189
     - 0.84790369
     - 0.99042435
     - 0.96746393
   * - 2000
     - 1.0125201
     - 0.84207406
     - 0.82677935
     - 0.99280963
     - 0.99867225
   * - 3000
     - 0.99927523
     - 0.83253672
     - 0.82242497
     - 0.99520066
     - 1.0081674
   * - 4000
     - 1.0243274
     - 0.84905774
     - 0.84131163
     - 0.99639833
     - 0.98169041

Balanced atomization energies (kJ/mol): janaf=1551.718000, nasa=1551.422031, atct_1.228=1551.598000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

The retained fit cites Dorofeeva 2003, so the JANAF-energy-only column is explicitly a **JANAF-proxy sensitivity**, not the physical correction to the retained source. The actual Dorofeeva Table 1 is separately replayed in ``adopted_source``: current/source ratios are 1.01839, 0.98895, 0.98179 and 1.00928 at the four listed temperatures. Its atomization energy on the explicit JANAF atom basis is 1551.798 kJ/mol and the ATcT-minus-source shift is -0.200 kJ/mol; the original FastChem fit atom basis remains incompletely recovered.

Current atmospheric footprint
-----------------------------

Nitric acid is relevant to atmospheric nitrogen reservoirs and proposed spectral signatures in cooler planetary atmospheres. The present hot reducing grid does not test those photochemical or condensed/aqueous roles.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.11151305e-22. Its largest participating-element share is 1.02795357e-18 (1.02795357e-16%) of N, at 4000 K, 100 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

No curated update is required by the available evidence. Optional future work may quantify the inherited fit residual or examine a complete modern torsional/rovibrational model above 3500 K. Any small ATcT energy sensitivity should use Dorofeeva physical energy and preserve the fit residual; do not substitute the older JANAF anchor.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `NIST-JANAF II: HNO3, H2SO4 and H2O2 (Dorofeeva et al.2003) <https://istina.msu.ru/media/publications/article/2c2/57d/1281832/2003_DOR_IOR_JPCRD_Three_molecules.pdf>`_; DOI 10.1063/1.1547435. Complete primary author-hosted paper inspected; Table1 selected numerical transcription pinned.
* `Partition sums for HITRAN2024 (Gamache et al., 2025) <https://www.homepages.ucl.ac.uk/~ucapsy0/pdf/25GaViTe.pdf>`_; DOI 10.1016/j.jqsrt.2025.109568. Complete author PDF and pinned official HAPI TIPS_2025 numerical arrays inspected.
* `ExoMol XI: spectrum of nitric acid (2015) <https://arxiv.org/abs/1507.02276>`_; DOI 10.1093/mnras/stv1376. Author abstract and stated500K applicability inspected; full line list not downloaded.

Replay with ``python mceg/reviews/hno3_source_review.py`` and
``pytest tests/model_evidence/hno3_source_review_test.py``.
The packet is ``tests/data/validation/hno3_source_review/20261106``;
the contract is ``mceg/contracts/hno3_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.

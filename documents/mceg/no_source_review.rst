NO: source review on the current curated baseline
=================================================

Prepare an ATcT energy-only candidate on the literal inherited fit, preserving its thermal function and fit residual. This is a smaller priority than NH2. A wholesale thermal replacement is not needed to correct the identified energy discrepancy: normalized XABC state sums closely reproduce NASA and the evaluated JANAF/NASA thermal difference is small over this range.

Identity and inspected sources
------------------------------

Neutral gas-phase nitric oxide NO, CAS 10102-43-9, X 2Pi ground electronic state with spin-orbit and Lambda-doubling structure; not NO+, NO- or the NO2/N2O molecular species.

The current source label is **JANAF 1998 (FastChem fit) [U]**. NASA CEA records cite Gurvich,1978,1989 pt1 p326 pt2 p203. The original numerical records and gas identities are pinned; shared ancestry is not independent validation.

ATcT 1.228 reduces balanced atomization energy by 1.024 kJ/mol relative to JANAF, whereas NASA already lies close to the modern energy. The diagnostic energy-only ratio is 0.921 at 1500 K and 0.970 at 4000 K. This is a physical D0 difference, not replacement of the fitted 1/T intercept. The direct XABC calculation supplies modern thermal evidence in addition to source-to-source agreement.

Qu et al. (2021) XABC and Gamache et al. (2024) were read in full. XABC supplies 30811 states through J=184.5; the downloaded file contains 21688 X-state levels, whereas the paper prose prints 21668 (the other manifold counts agree). The compact packet retains only energy, total weight, J and electronic label. Direct state moments reproduce the published partition nodes within 0.60 ppm; the small source/constants residual is retained rather than asserted to be exactly zero. The 2024 work uses these states and explicitly notes levels above the dissociation threshold, as well as isotope and entropy conventions. Its printed absolute natural-sample entropy should not be copied directly into a spin-free chemistry model: this review instead divides the ExoMol weights by the explicit nuclear-spin product 3 once, preserving both resolved spin-orbit ladders and Lambda components. No additional electronic degeneracy is multiplied in.

Source comparisons
------------------

All comparisons use atom formation from neutral ground-state atoms at one bar. Native NASA results preserve R=8.314510 J/mol/K; JANAF and the reaction engine use R=8.31446261815324. Molecular H is reconstructed from one fixed 298.15 K formation anchor plus the tabulated heat-content increment, never from the temperature-dependent formation-enthalpy column. NASA H(0) is its header Hf(298)-[H(298)-H(0)], not polynomial extrapolation to zero. No extra ZPE or well depth is introduced.

For 14N16O, nuclear-spin product=3 and rotational symmetry number=1. Total weights are exactly 3(2J+1); resolved electronic/fine-structure components already occur as distinct states. The lowest level is zero, no extra ZPE is added, and natural-mass translation is a declared principal-isotope proxy. A separate isotope-mass ratio is a sensitivity, not a balanced isotope reaction. All distributed states are used diagnostically; above-D0 and highest-J fractions do not prove missing-state completeness.

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
     - 1.0886376
     - 1.0831988
     - 0.99984919
     - 0.92117438
     - 0.91671047
   * - 2000
     - 1.0618847
     - 1.0616585
     - 0.99978036
     - 0.94027824
     - 0.94028448
   * - 3000
     - 1.0380911
     - 1.040256
     - 0.9994323
     - 0.9597783
     - 0.9623262
   * - 4000
     - 1.0303764
     - 1.0289944
     - 0.99855704
     - 0.96967945
     - 0.96977821

Balanced atomization energies (kJ/mol): janaf=627.835000, nasa=626.839972, atct_1.228=626.811000. ATcT uses version 1.228 and all participating atoms from that same version. Rounded marginal uncertainties are retained with the inputs; they do not provide reaction covariance or a complete uncertainty budget.

The XABC diagnostic uses 30811 numerical state rows (240 kB compressed), not transition lines. Spin-normalized Q/NASA rises only from 0.99998459 at 1500 K to 1.00040765 at 4000 K. The source-specific partition G/H/S/Cp, manifold fractions, above-threshold population and highest-J shell are replayed from those rows. This small thermal difference supports treating the energy correction separately; it does not prove complete physical accuracy.

Current atmospheric footprint
-----------------------------

NO is relevant to nitrogen chemistry and UV nightglow on Venus and Mars as discussed by XABC. These radiative/non-LTE applications are not validated by the hot hydrogen-rich equilibrium inventory.

The final **curated-tio2-v1** archive contains 75 converged ion-inclusive endpoints and 537 species: T=1500, 2000, 2500, 3000, 4000 K; P=1e-6, 1e-4, 0.01, 1, 100 bar; C/O=0.5888436553555908, 1, 1.5. The maximum mole fraction is 1.37656798e-07. Its largest participating-element share is 0.00184003952 (0.184003952%) of N, at 4000 K, 1 bar and C/O=0.588843655. Every participating element is accounted for in the packet. The archived target potential agrees with the literal current function; no new equilibrium solve was run.

These finite hot hydrogen-rich results do not bound cooler or oxidizing atmospheres, photochemistry, non-LTE populations, condensation or reaction kinetics. A small inventory here is not evidence that a molecule is unimportant in those settings.

Decision, access needs and MCEG
-------------------------------

Build a retained-thermal ATcT energy candidate and evaluate it on the current network before any adoption. If a later full XABC thermal representation is desired, validate continuous derivatives, bound/resonance policy, all isotope/spin conventions and compare against the 2024 thermodynamic tables. No unavailable paper is needed for the energy-only candidate or source-review merge.

This review changes neither curated data nor runtime behavior. It is independently mergeable under the established source-review standard: source attribution, normalized numerical comparisons and the bounded decision are supported; complete physical accuracy and complete historical ancestry remain unknown. Missing historical full texts are recorded in MCEG and are not a mandatory hold on this scoped review. Candidate construction and fixed-network substitution, if recommended, are a separate follow-up.

References and reproducibility
------------------------------

* `NIST-JANAF SRD 13 <https://doi.org/10.18434/T42S31>`_: official selected neutral-gas tables.
* `NASA CEA thermodynamic records <https://cearun.grc.nasa.gov/cea/thermo.inp>`_: NASA/TP-2002-211556; original cached full-file hash and selected exact records.
* `ATcT 1.228 <https://atct.anl.gov/Thermochemical%20Data/version%201.228/>`_: exact phase/CAS-selected formation enthalpies.
* `ExoMol XLII: XABC low-lying NO states (Qu et al., 2021) <https://academic.oup.com/mnras/article/504/4/5768/6273155>`_; DOI 10.1093/mnras/stab1154. Complete author PDF and official distributed states/partition file inspected.
* `Ideal Gas Thermodynamic Functions For NO From the Total Partition Sum and Its Moments (Gamache et al., 2024) <https://www.homepages.ucl.ac.uk/~ucapsy0/pdf/24GaOrQu.pdf>`_; DOI 10.1063/5.0209834. Complete author PDF inspected, including normalization and natural-sample sections.

Replay with ``python mceg/reviews/no_source_review.py`` and
``pytest tests/model_evidence/no_source_review_test.py``.
The packet is ``tests/data/validation/no_source_review/20261106``;
the contract is ``mceg/contracts/no_source_review.yaml``.
Source hashes resolve through the existing historical-byte resolver; output hashes refer directly to this packet. No mutable public-default alias or shared index is pinned.

H2O2 candidate: retain thermal functions and update evaluated energy
====================================================================

Recommend the balanced ATcT 1.228 energy-only candidate for a subsequent
adoption decision. It preserves the current Dorofeeva-based thermal function
and changes only the documented atomization-energy anchor. Both fixed-network
substitutions and independent solver checks pass. No production coefficients
are changed by this PR, and no new high-temperature partition model is adopted.

Construction and scientific boundaries
--------------------------------------

The actual FastChem4 H2O2 five-coefficient reaction is retained exactly.
ATcT 1.228 H=216.034, O=246.843 and HOOH=-129.400 kJ/mol Hf(0) give balanced
D0=1055.154 kJ/mol. The documented Dorofeeva-table-plus-JANAF-atom reference is
1055.542 kJ/mol. The candidate subtracts delta(D0)/(RT) from formation G/(RT),
where delta(D0)=-0.388 kJ/mol and R=8.31446261815324 J/mol/K. Formation G and H
therefore increase by 388 J/mol; S and Cp are unchanged at every temperature.
There is no new fit, internal join, spin correction or isotope correction.

The offset is anchored to the documented source evaluation. It does not assert
that the original finite-range FastChem fit uniquely determines D0 or that its
historical atom basis has been completely recovered. This is a controlled
energy update while retaining that representation, with the ambiguity stated.
The NASA record remains a source diagnostic rather than a second substituted
thermal model. Its native R is preserved in those diagnostics.

The prior full-text review of Dorofeeva (2003) supports retaining its thermal
evaluation. APTY's incomplete high-temperature state sum is not substituted;
its distribution stops at 2000 K. The improvement proposed here concerns the
modern evaluated central energy, not a claim that all H2O2 thermal uncertainty
has been removed. ATcT covariance is unavailable, so no synthetic confidence
interval is attached. See :doc:`h2o2_source_review` for source evidence and limits.

Actual finite substitutions
---------------------------

The saved curated-cos-v1 baselines contain 75 cases in each of the 537-species
ion-inclusive and 421-species neutral networks. All species, atom references,
element inventories and external conditions are fixed. The 150 baseline
endpoints are reused; 150 candidate endpoints, 12 tighter cold controls and
10 finite-path points are newly solved. Primary/tighter tolerances are 5e-14
and 2e-14. This run took about 19 seconds; that is an execution receipt rather
than a performance benchmark.

At 1500/3000/4000 K the isolated formation-K factors are
0.969369/0.984565/0.988401. The actual candidate/current mole-fraction ratio
across the grids is 0.969369--0.988401, with the small atomic-activity response
included. The maximum ion-inclusive H2O2 mole fraction changes from
3.18177379e-12 to 3.14486958e-12 at 4000 K, 100 bar, C/O=0.588844:
a decrease of 3.69042e-14. Counting both oxygen atoms, the oxygen inventory
fraction changes from 8.12378e-9 to 8.02956e-9. The neutral network agrees in
scale and sign. These figures do not describe photochemical peroxide formation
or cooler atmospheres.

The target response remains stable to 6.9e-13 in ln(mole fraction) under the
selected tighter cold controls, and finite mass-action closure is within
3.0e-13. The smooth five-point path is a real finite substitution. Reported
changes of other gases are extremely small: the largest raw water change is
about 6.17e-14 in mole fraction, while the largest all-species tolerance
control difference is 3.38e-14. The evidence supports the stable peroxide
response; it does not justify claiming an appreciable bulk-atmosphere change.

All endpoints converge and pass the archived element, charge and mass-action
criteria. The all-element relative criterion remains 1e-5, with tighter
species-element checks recorded separately, so trace-element numerical floors
are not hidden. Feasibility-corrected Gibbs ordering passes the 5e-14 tolerance;
the tiny species contribution does not establish a resolved physical Gibbs
improvement. Standalone FastChem independently reproduces selected baseline
and candidate equilibria with all log10(mole-fraction) errors below 1e-7.

Recommendation
--------------

Carry this energy-only candidate forward for adoption while retaining the
current thermal function. The old source's measured/evaluated energy and the
modern same-version atom balance are kept distinct; the proposed change is
small, continuous and scientifically better motivated than replacing the
thermal function with an incomplete state sum. Complete physical accuracy,
historical fit reconstruction and atmosphere-wide relevance remain unknown.

Reproducibility
---------------

``mceg/reviews/h2o2_candidate.py --build`` regenerates the exact candidate
and dense 2501-temperature validation. ``h2o2_equilibrium_impact.py`` runs
new fixed-network endpoints and controls with ``--output-dir`` and optional
``--fastchem-executable``. All archived production data remain unchanged.
The new candidate/impact packets retain their own receipts; the prior source
review and its hashes are preserved. MCEG records supported numerical claims
and a bounded recommendation, with full physical accuracy explicitly unknown.

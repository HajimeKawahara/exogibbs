H2O continuous thermal candidates (2026-11-04)
==============================================

This packet extends the dated 2026-11-03 source investigation without changing
its receipt or production coefficients. Both candidates represent H2(16O)
internal states with the aggregate NASA molecular mass 18.01528 u, 1 bar,
the existing NASA H/O thermal functions, and ATcT 1.172 total atomization
917.819 kJ/mol. This is an explicit dominant-isotope approximation.

``candidate.json`` contains the evaluated and discrete thermal models.
The evaluated model fits degree-8 Cp/R in scaled temperature to 251 rows
(1500 through 4000 K every 10 K) of the official 2025 thermochemical CSV.
Inverse published expanded-U(Cp) weights balance approximation error;
they do not assert independent measurement errors or a new uncertainty.
Analytic integrals define H and S using the 1500 K source values. Entropy
integration includes its logarithmic term; G=H-TS defines every derivative.
The numeric source H values are J/mol despite their original kJ/mol labels.
The source S already removes the nuclear-spin product four; only the
translation mass shift is applied to it.

``published_tables.npz`` freezes all 2501 integer-temperature rows from both
official CSVs on this domain. It standardizes the verified H unit and stores
S after the documented translation shift. Thus 2250 Cp rows and all H/S/Q
values except the two 1500 K anchors are withheld from fitting. The original
headers and representative unmodified rows remain in the historical source
packet. Printed rounding and expanded uncertainties are distinct from fit
error and are not physical-accuracy bounds for the complete reacting mixture.

The discrete model sums all 810270 supplied PW2024 states, using spin*(2J+1)
weights, Q/4, a shared para ground zero and no extra ZPE or symmetry divisor.
``discrete_nodes.json`` gives six exact state-moment nodes used for the C2
quintic Gibbs representation. ``discrete_off_knot_reference.json`` contains
45 additional direct sums withheld from that interpolation. No additional
scattering term is reconstructed or added. The small number of supplied
states above the first OH+H threshold are retained as in the source review.

``validation.json`` reports full-table and withheld-state comparisons,
analytic positive-Cp checks and evaluated/discrete thermal scope sensitivity.
The represented scope difference is not a new uncertainty distribution.
In particular, the 2016 scattering-density expression involves a phase-shift
derivative; separately representing OH and H does not alone prove simple
double counting. Assignment of such interaction contributions within the
ideal reacting-mixture model remains unverified.

Offline replay and optional reconstruction from pinned original files::

   PYTHONPATH=src python mceg/reviews/h2o_candidate.py
   PYTHONPATH=src pytest tests/model_evidence/h2o_candidate_test.py
   PYTHONPATH=src python mceg/reviews/h2o_candidate.py --extract /path/to/originals --output /tmp/h2o-rebuilt

The extractor verifies the three downloaded SHA256 values against the
historical source manifest before parsing. Full papers and original large
state files remain outside the packet. Official evaluated-source DOI:
10.1063/5.0273661, with versioned supplements listed in ``candidate.json``.
The separate ``h2o_equilibrium_impact/20261104`` packet measures conserved
finite substitutions on the unchanged current-default gas network.

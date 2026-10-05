# NASA9 references for the optional FastChem4 caloric model

`fastchem4_hot.json` is a **modified excerpt**, not an unmodified NASA data file.
It contains only the 1000–6000 K coefficient intervals for 28 elemental
references (27 neutral gas atoms and the electron) and 34 explicitly matched gas
species. Numerical coefficients are unchanged. Record names, source line
numbers, original references, elemental compositions, and atomic weights are
retained in each record.

Source: NASA's official [`data/thermo.inp`](https://github.com/nasa/cea/blob/2f79a647e85737742fb08a127da00d1ea1d069da/data/thermo.inp),
commit `2f79a647e85737742fb08a127da00d1ea1d069da`. The JSON records the SHA-256 of
the complete downloaded source, including its original line endings. The
accompanying Apache 2.0 license and NASA NOTICE apply to this excerpt.

The NASA9 definitions and standard states are described in
[McBride, Zehe & Gordon (2002), NASA/TP-2002-211556](https://ntrs.nasa.gov/citations/20020085330).
Pressure is 1 bar. Coefficients give dimensionless `cp/R`, `h/RT`, and `s/R`;
the consuming model uses a single common molar gas constant. These are
natural-element thermochemical data and tabulated atomic weights, not a
resolved isotopologue model. Historical constants introduce small differences
from statistical functions evaluated using current SI constants.

The optional model combines these data with the already packaged FastChem4
`logK.dat` (SHA-256
`eda9b9d7e62ccf7398036ee84349bd0a721b779acdc9d191cbf9fe1dd01f619c`).
An elemental reference transformation restores absolute standard Gibbs
functions and entropy to the reaction fits. Using reaction log K alone as an
absolute Gibbs function would omit elemental heat capacities.

An audit of the **539-species**, ion-inclusive FastChem4 gas network using the
default elemental set found nine organic
reaction fits with negative reconstructed heat capacities within 1500–4500 K.
Additional dimers and atomic ions fell below the ideal-gas translational
`cp/R = 2.5` limit beyond reasonable polynomial fit error. The explicit
`replacements` mapping selects 34 chemically identified NASA records. Both
their chemical potentials and caloric functions are replaced together;
ordinary FastChem chemistry cannot be combined with the replacement entropy.
Molecular identity was checked against the descriptive names and source
references, not only the elemental formula. In particular, glyoxal uses its
named NASA glyoxal record, not an arbitrary C2H2O2 isomer.

**Diacetyl peroxide (`C4H6O4`) is excluded**, because it has an invalid
high-temperature reaction fit and no matching NASA record in this source.
Acetic-acid dimers are not an acceptable substitute. The resulting opt-in
model has **538 species**, and preserves all 28 conservation rows. The original
FastChem4 preset and data remain unchanged. This is an explicit approximate
thermochemical model, not complete validation of the original 539-species table
or of every original source's high-temperature measurements.

Evaluation is restricted to **1500–4500 K**, inside one smooth NASA interval.
No coefficient switching, clipping, or extrapolation is used. Each retained
species' heat-capacity minimum is checked at both endpoints and all interior
stationary points. The minimum is `cp/R = 2.4903374169769754`; a tolerance of
0.01 below 2.5 accounts for fit discrepancies (including small departures in
the NASA atomic-ion fits). No heat capacity is altered to pass this check.
Temperatures outside the declared domain return NaN, and setup rejects an
extended domain. Positive species heat capacities and ideal-gas equilibrium
imply positive equilibrium heat capacity at a stable, nonsingular solution.
This check does not replace equilibrium convergence, conservation, or domain
checks, nor establish the accuracy of a WASP-18b retrieval.

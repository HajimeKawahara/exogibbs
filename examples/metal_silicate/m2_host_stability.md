# Native competing phases at the supplied BSE host

`m2_host_stability.py` evaluates the actual final MELTS host against the
native alphaMELTS candidate catalog. It extends the existing
[alloy insertion controls](m2_phase_selection.md) with an explicit mineral
assessment. It does not resolve the archived source or certify a stable
liquid, and does not change `select_metal_phase` acceptance.

ExoEOS owns the optional `evaluate_liquid(..., include_saturation=True)`
property calculation. A fresh pinned worker calls `calcSaturationState` with
the supplied liquid oxide amounts and no imposed oxygen buffer. Every native
candidate stays in the receipt, including missing affinities. The worker
evaluates each available incipient composition with `calcPhaseProperties`
and checks its returned mass and oxide amounts against the requested 100 g
basis. Signed oxide amounts are retained: native Fe metal is represented by
a combination of FeO and negative Fe2O3. A failed candidate round trip is
unresolved; a native candidate failure is saved and later candidates are
marked unevaluated so they cannot use potentially corrupted state. A failed
liquid or saturation calculation still aborts the worker. No equilibrium or
liquidus search is called.

The [MELTS manual](https://melts.ofm-research.org/Manual/UnixManHtml/Solid-Composition.html)
defines a smaller native affinity as closer to saturation. Its
[phase-list guidance](https://melts.ofm-research.org/Applet/helpHtml/helpMELTSgui.html)
also distinguishes an unavailable estimate from phase absence and notes
that incipient solution compositions far from saturation are approximate.
Raw affinities are recorded in the backend's J convention; their magnitudes
are not compared across different formula normalizations. Insertion energies
are independently computed from extensive candidate energies on an explicit
amount basis, then divided by the candidate's total moles of atoms.

For a candidate with Gibbs energy `G_c` and native host component vector `c`,
the trial consumes `epsilon*c` from the host and adds `epsilon` times that
candidate. Negative entries in `c` produce host components and are allowed.
Every consumed component must be present. Even a tiny positive coefficient
cannot consume an exactly absent component; unavailable endpoint potentials
also remain unresolved. The maximum feasible step and the oxide reconstruction
error are recorded without clipping or rounding coefficients to force support.

The extensive scalar already augments native MELTS with ideal dissolved H2.
If `N` is the sum of native host component moles and `h` the molecular H2
amount, the host potentials are

```text
mu_augmented / RT = mu_native / RT + ln(N/(N+h))
delta_G_trial / RT = G_c / RT - c . (mu_native / RT)
                     - sum(c) * ln(N/(N+h))
```

Native water stays in MELTS; no second water-solubility relation is added.
This correction is essential even though the added H2 amount stays fixed
during the trial. The candidate composition is the *native* incipient estimate;
it is not reoptimized after adding this correction. The calculation is
invariant under a common elemental energy gauge and a rescaling of the host
and H2 amounts. Tests also differentiate the full ideal-H2 dilution scalar
along an independent feasible insertion path.

A sufficiently negative feasible trial can reject the host within these
formal phase models. Nonnegative trials remain `unresolved`: neither a
global search over each solution phase, liquid splitting, nor empirical
model applicability has been established. Native Fe-Ni solid/liquid alloys
and pure water are identified separately from the 30 mineral candidates.
They are alternative constitutive models, not a substitution for the selected
Ma Fe-Si-O-H alloy or the common atmospheric gas/cloud catalog. The native
33-phase catalog is distinct from the atmospheric 26 condensate candidates.

## Optional composition and liquid-splitting searches

`m2_stability_search.py` adds bounded searches for negative counterexamples.
An updated ExoEOS evaluator supplies native endmember oxide bases and direct
`calcMolarProperties` energies. This bypasses inverse oxide conversions that
can change hornblende or orthoamphibole amounts; the original conversion
failure remains in the property receipt. Kalsilite stays in the catalog:
its incipient estimate can be unavailable even when interior compositions
have finite native energies.

`search_competing_solutions` searches the nonnegative native endmember
simplex. Every composition uses the same insertion calculation and ideal-H2
dilution correction above. This simplex is an explicit search subset, not an
empirically calibrated domain or necessarily the full native solution domain.
`search_liquid_splitting` partitions every positive parent component, including
molecular H2, into two finite daughter liquids. Daughter component amounts
sum to the parent; exact-zero parent components stay zero. The same native
energy and ideal-H2 scalar are evaluated for each daughter. Any identical
linear H2 standard, including a declared sensitivity offset, cancels in the
parent-minus-daughters comparison. Native H2O is not replaced or double counted.

Both searches save evaluated points, failures, optimizer messages, the budget,
and a fresh best-point evaluation. A negative feasible witness rejects this
particular supplied host under the declared formal model. Nonnegative points
remain `unresolved`, with `lower_bound_rt=None` and no minimum certificate.
Neither a successful optimizer nor exhausted starts establish absence.
Liquid splits search daughter fractions from 0.001 to 0.999; these are search
bounds, not a trace floor in the source model. Unsearched endpoints, additional
liquids, native domains outside the simplex, and model calibration remain open.

Append `--search-solutions hornblende orthoamphibole kalsilite` and/or
`--search-liquid-splitting` to the command below. `--search-evaluations 60`
limits objective calls per search, followed by one fresh final evaluation.
No search runs by default. These postprocessing trials retain fixed source
T/P; they are not a new planetary closure or a comparison of Gibbs energies
at different bottom pressures.

Run a new assessment of either a provider contact archive or an Inventory
column archive with the matching optional ExoEOS checkout:

```sh
JAX_ENABLE_X64=1 PYTHONPATH=src \
python examples/metal_silicate/run_m2_host_stability.py \
  --source-json /path/to/recorded-contact-or-column.json \
  --exoeos-checkout /path/to/exoeos \
  --runtime /path/to/alphamelts-py-2.3.2-ubuntu_22_04-x86_64 \
  --python /path/to/melts-env/bin/python \
  --output /tmp/new-host-stability.json
```

The output must not exist. It records the input SHA256, unchanged source
amounts, assessment and provider code hashes, runtime hashes, native property
receipt, candidate catalog, and unresolved reasons. An assessment of an old
source is a new property evaluation of that source's composition; it does
not revise the provenance or scientific status of the original equilibrium.

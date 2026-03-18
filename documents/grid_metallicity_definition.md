# Note on the composition-axis meaning for equilibrium grids

The current equilibrium-grid implementation uses a composition axis labeled as `log10(Z/Zsun)`. However, the present grid-generation rule does **not** yet exactly correspond to the physical metallicity defined from mass fractions.

At the moment, the forward grid builder constructs the elemental abundance vector by:

- keeping H and He fixed at the reference abundance pattern,
- scaling all non-H, non-He, non-electron elements uniformly by a factor `10**m`,
- setting electrons to zero.

Therefore, the current axis parameter `m` is, strictly speaking, a **uniform metal scaling factor in number-abundance space relative to the reference composition**, not the physical

\[
\log_{10}(Z/Z_\odot)
\]

defined from mass fractions.

This distinction matters because the physical metallicity uses

\[
X + Y + Z = 1
\]

with

- \(X\): hydrogen mass fraction
- \(Y\): helium mass fraction
- \(Z\): total mass fraction of all heavier elements

whereas the current implementation scales metal number abundances directly and leaves H and He unchanged. In general, these two parameterizations are not exactly identical.

## Physical definition to adopt

The intended final meaning of the composition axis is the physical metallicity

\[
\log_{10}(Z/Z_\odot)
\]

where \(Z\) is computed from the elemental number-abundance vector \(b\) using elemental masses \(A_i\):

\[
Z(b) =
\frac{\sum_{i \notin \{\mathrm H,\mathrm{He},e^-\}} A_i b_i}
{\sum_{i \in \mathrm{all\ elements\ except}\ e^-} A_i b_i}
\]

Here:

- \(b_i\) is the elemental abundance in number units,
- \(A_i\) is the elemental mass, obtained from `utils/elements.py::element_mass`,
- electrons are ignored in the mass sum.

The solar-reference metallicity \(Z_\odot\) is computed from the reference elemental abundance vector in the same way:

\[
Z_\odot = Z(b_{\rm ref})
\]

and the composition-axis value is then

\[
\log_{10}(Z/Z_\odot)
=
\log_{10}\!\left(\frac{Z(b)}{Z(b_{\rm ref})}\right)
\]

## Consequence for the implementation

Because the current forward grid builder uses a metal-scaling parameter rather than the physical mass-fraction metallicity, the forward and inverse mappings are not yet fully consistent.

To make the grid axis physically correct, the code should be updated so that:

1. the grid-generation input parameter is interpreted as physical `log10(Z/Zsun)`,
2. this target mass-fraction metallicity is converted into the corresponding elemental abundance vector under the adopted H/He-atmosphere convention,
3. the inverse path from `b` to `log10(Z/Zsun)` uses the same physical definition.

## Planned transition

The next implementation steps should therefore be:

1. add a helper to compute physical \(Z\) from an elemental abundance vector using `element_mass`,
2. compute \(Z_\odot\) from `setup.element_vector_reference`,
3. update the forward grid-construction path so that the stored composition axis truly represents physical `log10(Z/Zsun)`,
4. then implement the inverse mapping from `b` to physical `log10(Z/Zsun)` using the same convention.

Until that transition is completed, the current composition-axis label should be understood as an approximation to physical metallicity, not yet the exact mass-fraction definition.
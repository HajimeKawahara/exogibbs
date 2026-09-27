# Global insertion for the associated alloy with H–O interaction

`m2_associated_global.make_associated_insertion_minimizer(metadata,
metal_evaluator, exoeos_checkout)` supplies an optional source-phase insertion
minimizer. Pass the returned callable to
`phase_selection.select_metal_phase(metal_insertion_minimizer=...)`.
Its arguments follow the existing source convention: temperature in K,
pressure in bar, the full element-by-species formula matrix, the actual
elemental potentials, and the effective closed composition domain.

The physical scalar, standards, interactions and reference curvature remain
ExoEOS-owned. The factory verifies the saved provider hashes, component and
element order, interactions, and selected 18- or 19-species model. Independent
linear scenario shifts remain separate from the provider's base standards.
The actual source callback is evaluated again at the final candidate. Its
scalar and every free reduced chemical-potential direction must agree with
the independently evaluated saved expression. Checking the energy alone would
miss a changed linear standard orthogonal to that single composition.

Let `G0` be the same declared alloy with only its additional H–O interaction
omitted. The provider must establish a positive curvature lower bound `kappa`
for `G0` throughout the original numerical domain. For the 18-species model,
`G - G0 = epsilon*x_O*x_H`. Adding `epsilon` to the O/H Hessian diagonal makes
this perturbation positive semidefinite. For the 19-species K perspective,
the perturbation is `epsilon*x_O*x_H/(1-x_K)`. Outward Gershgorin row sums give
safe diagonal additions only in O/H/K; its nonnegative K/K term may be dropped
conservatively. Neither model inherits positive curvature for its full H–O
scalar, and the domain is never shrunk to manufacture convexity.

For each closed box, the alphaBB term
`sum(rho_i*(x_i-lo_i)*(x_i-hi_i))/2` is nonpositive and has the required diagonal
Hessian. The resulting minorant retains the reference curvature. At a rational
anchor its outward value and gradient give a separable quadratic supporting
bound. Each one-dimensional quadratic is minimized exactly using rational
candidates and outward arithmetic. This bound relaxes, rather than removes,
any dependent-coordinate constraint. The implementation currently requires
the entire independent box to satisfy the declared Fe limits.

Only O/H (and K if present) are bisected. Every other composition remains in
the full original interval at every node. Zero amounts are retained through
the continuous ideal-mixing limit; positive numerical search coordinates
are anchors, never new physical lower bounds. Numerical optimization is not
an acceptance condition. The global frontier lower bound and a feasible
outward upper bound must differ by at most the requested tolerance, normally
`1e-8 RT` per chemical-species mole. Budget exhaustion remains unresolved.

`InsertionMinimum.global_certificate` preserves the complete retained
frontier, exact candidate composition, outward bounds, reference curvature,
convexification coefficients, actual T/P/elemental plane, and provider and
verifier file hashes. Existing source acceptance, complementarity, finite
atom closure, numerical-bound-contact and host-stability gates remain in
force. This mathematical minimum certificate is not an empirical calibration
of the H–O interaction or the associated melt.

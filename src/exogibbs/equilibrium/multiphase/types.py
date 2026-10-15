"""Static physical contracts and array results for local multiphase solves."""

from dataclasses import dataclass, fields
from typing import Any, Callable, Tuple

import jax
import numpy as np


GibbsFunction = Callable[[Any, Any, Any, Any], Any]
R_SCALE = 8.31446261815324


def _names(values, label):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of unique names.")
    values = tuple(values)
    if (not values or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{label} must contain nonempty, unique names.")
    return values


def _matrix(values, columns, label, *, empty=False):
    array = np.asarray(values, dtype=float)
    if empty and array.size == 0:
        return ()
    if (array.ndim != 2 or array.shape[1] != columns
            or not np.all(np.isfinite(array))):
        raise ValueError(f"{label} must be a finite matrix with {columns} columns.")
    return tuple(tuple(row) for row in array)


@dataclass(frozen=True)
class Phase:
    """A nonnegative component basis and full extensive G(T_K, P_Pa, n_mol, params).

    Atom-matrix rows follow Problem.elements. Optional homogeneous domain
    rows mean C @ n >= 0; they are physical restrictions, not amount floors.
    The callback includes standards, mixing, excess and pressure terms once.
    It must support JAX derivatives on the selected component face. All phases
    receive the same explicit dynamic parameters PyTree.

    An optional insertion_lower_bound_j_mol(T, P, element_dual, parameters,
    allowed_components) returns a global lower bound on G(x)-dual @ A @ x
    in J/mol components, on sum(x)=1 and the declared domain. The last argument
    is a static tuple of indices allowed by the element budget. NaN means
    unavailable. The provider/caller owns the validity of this bound.

    Optional derivative_domain(T, P, n, parameters) is a JAX scalar predicate
    for a smooth material branch. Without it the caller asserts smoothness
    on the selected face. An absent solution's lower-bound callback must also
    support JAX when used by solve_fixed_support's derivative audit.
    """

    name: str
    components: Tuple[str, ...]
    atom_matrix: Any
    gibbs_j: GibbsFunction
    domain_matrix: Any = ()
    domain: str = "nonnegative component amounts"
    standard_state: str = "caller supplied"
    insertion_lower_bound_j_mol: Any = None
    derivative_domain: Any = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Phase name must be nonempty.")
        object.__setattr__(self, "components", _names(self.components, "components"))
        count = len(self.components)
        atoms = _matrix(self.atom_matrix, count, "atom_matrix")
        if np.any(np.asarray(atoms) < 0) or np.any(np.sum(atoms, axis=0) <= 0):
            raise ValueError("Every component needs nonnegative atom counts and atoms.")
        object.__setattr__(self, "atom_matrix", atoms)
        object.__setattr__(self, "domain_matrix", _matrix(
            self.domain_matrix, count, "domain_matrix", empty=True,
        ))
        if not callable(self.gibbs_j):
            raise ValueError("gibbs_j must be a callable extensive scalar.")
        if (self.insertion_lower_bound_j_mol is not None
                and not callable(self.insertion_lower_bound_j_mol)):
            raise ValueError("insertion_lower_bound_j_mol must be callable or None.")
        if self.derivative_domain is not None and not callable(self.derivative_domain):
            raise ValueError("derivative_domain must be callable or None.")
        if not self.domain or not self.standard_state:
            raise ValueError("Declare the phase domain and standard state.")


@dataclass(frozen=True)
class Problem:
    """Ordered phases sharing an explicit mol-atoms elemental ledger.

    Charge is not an element; charge constraints and signed formal endmember
    coordinates are outside this first contract.
    """

    elements: Tuple[str, ...]
    phases: Tuple[Phase, ...]

    def __post_init__(self):
        object.__setattr__(self, "elements", _names(self.elements, "elements"))
        object.__setattr__(self, "phases", tuple(self.phases))
        _names(tuple(p.name for p in self.phases), "phase names")
        if any(len(p.atom_matrix) != len(self.elements) for p in self.phases):
            raise ValueError("Atom-matrix rows must follow Problem.elements.")

    @property
    def component_slices(self):
        """Slices in phase order; names are local to each phase."""
        offsets = np.cumsum([0] + [len(p.components) for p in self.phases])
        return tuple(slice(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:]))

    @property
    def atom_matrix(self):
        return np.concatenate([p.atom_matrix for p in self.phases], axis=1)

    @property
    def domain_matrix(self):
        matrix = np.zeros((sum(len(p.domain_matrix) for p in self.phases),
                           self.atom_matrix.shape[1]))
        offset = 0
        for phase, section in zip(self.phases, self.component_slices):
            count = len(phase.domain_matrix)
            if count:
                matrix[offset:offset + count, section] = phase.domain_matrix
            offset += count
        return matrix


@dataclass(frozen=True)
class Support:
    """Static positive component indices and active C-row indices.

    All other components are exactly zero. Constraint indices follow the
    concatenation of Phase.domain_matrix rows in Problem phase order.
    This declaration is not evidence for absence of excluded phases.
    """

    components: Tuple[int, ...]
    active_constraints: Tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("components", "active_constraints"):
            values = tuple(getattr(self, name))
            if (any(not isinstance(i, (int, np.integer)) or i < 0 for i in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{name} must contain unique nonnegative indices.")
            object.__setattr__(self, name, values)
        if not self.components:
            raise ValueError("Support must contain at least one component.")


@dataclass(frozen=True)
class Options:
    """Dimensionless numerical tolerances; use JAX x64 for these defaults."""

    max_iterations: int = 100
    max_backtracks: int = 50
    stationarity_tolerance: float = 1e-10
    conservation_tolerance: float = 1e-9
    kkt_tolerance: float = 1e-8
    curvature_tolerance: float = 1e-9
    support_tolerance: float = 1e-12
    max_supports: int = 32
    derivative_rcond: float = 1e-10

    def __post_init__(self):
        integers = ("max_iterations", "max_backtracks", "max_supports")
        for name in integers:
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for field in fields(self):
            if field.name in integers:
                continue
            value = getattr(self, field.name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be positive and finite.")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Result:
    """Absolute state and independently recomputed fixed-support diagnostics.

    KKT and minimum checks concern only the declared face, not insertion in
    omitted components or phases.
    Residuals and curvature use G/(R_SCALE*T*sum(b)) and q=n/sum(b).
    Undefined chemical potentials and unevaluated insertion bounds are NaN.
    Element potentials use a row-scaled minimum-norm gauge if elemental rows
    are redundant; rows absent from the support are undefined. minimum_valid
    requires strict positive tangent curvature (or no free directions), not
    just semidefiniteness. The host insertion audit reports global stability
    separately. derivative_valid is checked by solve_fixed_support; the host
    search/proof entry points do not provide transformed derivatives.
    """

    amounts_mol: Any
    phase_amounts_mol: Any
    phase_element_amounts_mol: Any
    gibbs_j: Any
    chemical_potentials_j_mol: Any
    element_potentials_j_mol: Any
    constraint_multipliers_j_mol: Any
    conservation_residual_mol: Any
    conservation_error: Any
    stationarity_error: Any
    domain_slack: Any
    complementarity_error: Any
    minimum_curvature: Any
    iterations: Any
    converged: Any
    local_kkt_valid: Any
    minimum_valid: Any
    support_valid: Any
    derivative_valid: Any
    dual_unique: Any
    line_search_failed: Any
    insertion_lower: Any
    insertion_upper: Any
    incipient_compositions: Any
    support: Support
    stability_status: str = "unchecked"

    def tree_flatten(self):
        children = tuple(getattr(self, f.name) for f in fields(self)[:-2])
        return children, (self.support, self.stability_status)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return cls(*children, *auxiliary)

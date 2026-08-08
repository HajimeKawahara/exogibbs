"""Local fixed-support condensation retrieval with ExoJAX and NUTS.

This example uses the FastChem4 gas tables with a carbon-rich ``C/O = 2``
elemental inventory and a deliberately reduced condensate catalog containing
only FastChem4's graphite phase, ``C(s)``.  A nominal gas-only profile is
solved before sampling, graphite stability is inspected, and the layers with
a positive graphite solution are certified with the differentiable
fixed-support zero-barrier kernel.  Those layer indices and all numerical
initial values are then frozen.  Inside NUTS, active layers use the condensate
custom VJP and the remaining layers use the gas-only custom VJP.

The fixed mask is a deliberately local inference contract.  The script checks
all corners of its narrow prior box before sampling and stops if a corner
changes graphite support or fails a primal solve.  It does not differentiate
support discovery, the production condensate lifecycle, phase transitions, or
rainout.  It also makes no phase-stability or support-closure claim for the
omitted FastChem4 condensates.  In particular, widening the priors or treating
this reduced model as full-catalog equilibrium is not supported.

Run a chemistry-only check (no ExoJAX database is opened) with::

    python examples/retrievals/exojax_nuts_condensate_fixed_support.py \
        --preflight-only --nlayer 8

Run a short end-to-end retrieval with a pre-existing ExoMol CO database with::

    python examples/retrievals/exojax_nuts_condensate_fixed_support.py \
        --quick --co-database .database/CO/12C-16O/Li2015

The full run is intended for a GPU.  This example never downloads a database.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from _exojax_nuts_common import (  # noqa: E402
    TRUTH_ALPHA,
    TRUTH_LOG_CO_SCALE,
    TRUTH_T0_K,
    add_common_cli_arguments,
    build_spectral_context,
    co_emission_flux,
    make_mock_observation,
    resolve_demo_shape,
    resolve_run_settings,
    run_reverse_mode_nuts,
    write_run_outputs,
)
from exogibbs.api.gas import (  # noqa: E402
    EquilibriumOptions,
    solve_profile as solve_gas_profile,
)
from exogibbs.equilibrium.condensate.acceptance import (  # noqa: E402
    least_squares_element_potential,
)
from exogibbs.equilibrium.condensate.setup import (  # noqa: E402
    build_condensate_chemical_setup,
)
from exogibbs.equilibrium.condensate.fixed_support import (  # noqa: E402
    minimize_gibbs_fixed_support,
    minimize_gibbs_fixed_support_with_diagnostics,
)
from exogibbs.equilibrium.gas.kernel.diagnostics import (  # noqa: E402
    minimize_gibbs_with_diagnostics,
)
from exogibbs.equilibrium.gas.kernel.solver import minimize_gibbs  # noqa: E402
from exogibbs.equilibrium.gas.types import ThermoState  # noqa: E402
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup  # noqa: E402
from exogibbs.thermo.models import ChemicalSetup  # noqa: E402


jax.config.update("jax_enable_x64", True)

CASE_NAME = "condensate_fixed_support"
GRAPHITE_SPECIES = "C(s)"
CARBON_TO_OXYGEN_RATIO = 2.0
REFERENCE_PRESSURE_BAR = 1.0
PRESSURE_TOP_BAR = 1.0e-3
PRESSURE_BOTTOM_BAR = 10.0

# This box is intentionally narrow.  The eight corners are checked before NUTS.
T0_PRIOR_BOUNDS_K = (1155.0, 1165.0)
ALPHA_PRIOR_BOUNDS = (0.029, 0.031)
LOG_CO_SCALE_PRIOR_BOUNDS = (-0.005, 0.005)

GAS_RESIDUAL_TOLERANCE = 1.0e-11
GAS_MAX_ITERATIONS = 1000
CONDENSATE_RESIDUAL_TOLERANCE = 1.0e-10
CONDENSATE_MAX_ITERATIONS = 100
GRAPHITE_SEED_FRACTION = 1.0e-3
MIN_ACTIVE_GRAPHITE_AMOUNT = 1.0e-8
MIN_INACTIVE_DRIVING_MARGIN = 2.0e-2
DEFAULT_RELATIVE_NOISE = 2.0e-3
CONDENSATE_MODEL_SCOPE = "fastchem4_gas_plus_graphite_only"


@dataclass(frozen=True)
class GraphiteProfilePlan:
    """Static layer partition and initial values used inside NUTS."""

    setup: Any
    pressures_bar: Any
    nominal_temperatures: Any
    reference_element_vector: Any
    carbon_index: int
    oxygen_index: int
    co_species_index: int
    graphite_species_index: int
    active_indices: tuple[int, ...]
    inactive_indices: tuple[int, ...]
    gas_only_log_amounts: Any
    gas_only_total_log_amounts: Any
    hybrid_log_amounts_init: Any
    hybrid_total_log_amounts_init: Any
    graphite_amounts_init: Any
    graphite_seed_amount: float
    nominal_graphite_driving_margin: Any
    nominal_fixed_support_residual: Any

    @property
    def active_mask(self) -> np.ndarray:
        """Return the fixed graphite mask as a host NumPy array."""

        mask = np.zeros((len(self.pressures_bar),), dtype=bool)
        mask[np.asarray(self.active_indices, dtype=int)] = True
        return mask


def pressure_profile(
    nlayer: int,
    *,
    pressure_top: float = PRESSURE_TOP_BAR,
    pressure_bottom: float = PRESSURE_BOTTOM_BAR,
) -> jax.Array:
    """Return ExoJAX's endpoint-inclusive top-to-bottom pressure grid."""

    if nlayer < 2:
        raise ValueError("nlayer must be at least two for the hybrid demo.")
    if pressure_top <= 0.0 or pressure_bottom <= pressure_top:
        raise ValueError("pressure bounds must satisfy 0 < top < bottom.")
    return jnp.logspace(
        jnp.log10(pressure_top),
        jnp.log10(pressure_bottom),
        nlayer,
        dtype=jnp.float64,
    )


def powerlaw_temperature(
    pressures_bar: Any,
    t0_kelvin: Any,
    alpha: Any,
) -> jax.Array:
    """Use the same one-bar power-law convention as the ExoJAX atmosphere."""

    pressure = jnp.asarray(pressures_bar, dtype=jnp.float64)
    return jnp.asarray(t0_kelvin) * pressure ** jnp.asarray(alpha)


def graphite_only_chemical_setup(
    full_setup: Optional[Any] = None,
):
    """Build the declared FastChem4-gas plus graphite-only chemistry model."""

    source = (
        condensate_chemical_setup(silent=True)
        if full_setup is None
        else full_setup
    )
    graphite_index = source.condensate_species.index(GRAPHITE_SPECIES)
    source_condensates = source.condensate_setup
    selected_index = jnp.asarray([graphite_index], dtype=jnp.int32)

    def graphite_hvector(temperature):
        return jnp.take(
            source_condensates.hvector_func(temperature),
            selected_index,
            axis=-1,
        )

    validity = source_condensates.temperature_validity_upper
    selected_validity = None
    if validity is not None:
        selected_validity = (float(validity[graphite_index]),)
    metadata = dict(source_condensates.metadata or {})
    source_catalog_count = len(source.condensate_species)
    metadata.update(
        {
            "model_scope": CONDENSATE_MODEL_SCOPE,
            "condensate_catalog_mode": "reduced_explicit",
            "reduced_condensate_catalog": True,
            "selected_condensates": (GRAPHITE_SPECIES,),
            "condensate_species_considered": (GRAPHITE_SPECIES,),
            "source_condensate_catalog_count": source_catalog_count,
            "full_fastchem4_condensate_species_count": source_catalog_count,
            "excluded_condensate_species_count": source_catalog_count - 1,
            "full_catalog_equilibrium_claimed": False,
            "full_catalog_support_closure_checked": False,
            "rainout": False,
            "temperature_validity_upper": selected_validity,
        }
    )
    reduced_condensates = ChemicalSetup(
        formula_matrix=source.formula_matrix_cond[:, selected_index],
        hvector_func=graphite_hvector,
        elements=tuple(source.elements),
        species=(GRAPHITE_SPECIES,),
        element_vector_reference=source_condensates.element_vector_reference,
        metadata=metadata,
        temperature_validity_upper=selected_validity,
    )
    return build_condensate_chemical_setup(
        gas_setup=source.gas_setup,
        condensate_setup=reduced_condensates,
    )


def carbon_rich_reference_vector(setup: Any) -> jax.Array:
    """Return the FastChem4 reference vector with C/O set exactly to two."""

    if setup.gas_setup.element_vector_reference is None:
        raise ValueError("FastChem4 setup did not provide a reference inventory.")
    carbon_index = setup.elements.index("C")
    oxygen_index = setup.elements.index("O")
    element_vector = jnp.asarray(
        setup.gas_setup.element_vector_reference,
        dtype=jnp.float64,
    )
    return element_vector.at[carbon_index].set(
        CARBON_TO_OXYGEN_RATIO * element_vector[oxygen_index]
    )


def scale_carbon_and_oxygen(
    element_vector: Any,
    carbon_index: int,
    oxygen_index: int,
    log_co_scale: Any,
) -> jax.Array:
    """Scale C and O together while preserving the carbon-rich C/O ratio."""

    values = jnp.asarray(element_vector)
    indices = jnp.asarray([carbon_index, oxygen_index], dtype=jnp.int32)
    scale = jnp.power(jnp.asarray(10.0, dtype=values.dtype), log_co_scale)
    return values.at[indices].set(values[indices] * scale)


def _graphite_support_arrays(setup: Any, graphite_index: int):
    support = jnp.asarray([graphite_index], dtype=jnp.int32)
    formula_matrix_cond = setup.formula_matrix_cond[:, support]

    def graphite_hvector(temperature):
        return setup.condensate_setup.hvector_func(temperature)[support]

    return support, formula_matrix_cond, graphite_hvector


def _graphite_inactive_margin(
    setup: Any,
    graphite_index: int,
    temperature: Any,
    pressure_bar: Any,
    gas_log_amounts: Any,
    total_gas_log_amount: Any,
) -> jax.Array:
    """Return ``h_C(s) - A_C(s)^T lambda`` for a gas-only state."""

    gas_source = (
        setup.gas_setup.hvector_func(temperature)
        + jnp.log(pressure_bar / REFERENCE_PRESSURE_BAR)
        - total_gas_log_amount
    )
    potential = least_squares_element_potential(
        formula_matrix=setup.formula_matrix,
        gas_ln_n=gas_log_amounts,
        gas_stationarity_source=gas_source,
    )
    condensate_column = setup.formula_matrix_cond[:, graphite_index]
    return (
        setup.condensate_setup.hvector_func(temperature)[graphite_index]
        - condensate_column @ potential
    )


def _audit_graphite_candidate(
    setup: Any,
    graphite_index: int,
    temperatures: Any,
    pressures_bar: Any,
    element_vectors: Any,
    gas_log_amounts_init: Any,
    gas_total_log_amounts_init: Any,
    graphite_amounts_init: Any,
    fixed_gas_log_amounts_init: Optional[Any] = None,
    fixed_total_log_amounts_init: Optional[Any] = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Audit gas roots, graphite driving, and fixed-support roots by layer."""

    _, formula_matrix_cond, graphite_hvector = _graphite_support_arrays(
        setup, graphite_index
    )

    def gas_layer(temperature, pressure, inventory, q_init, qtot_init):
        return minimize_gibbs_with_diagnostics(
            ThermoState(
                temperature,
                jnp.log(pressure / REFERENCE_PRESSURE_BAR),
                inventory,
            ),
            q_init,
            qtot_init,
            setup.formula_matrix,
            setup.gas_setup.hvector_func,
            epsilon_crit=GAS_RESIDUAL_TOLERANCE,
            max_iter=GAS_MAX_ITERATIONS,
        )

    gas_log_amounts, gas_diagnostics = jax.vmap(gas_layer)(
        temperatures,
        pressures_bar,
        element_vectors,
        gas_log_amounts_init,
        gas_total_log_amounts_init,
    )
    gas_total_log_amounts = jax.scipy.special.logsumexp(
        gas_log_amounts, axis=1
    )
    margins = jax.vmap(
        lambda temperature, pressure, q, qtot: _graphite_inactive_margin(
            setup,
            graphite_index,
            temperature,
            pressure,
            q,
            qtot,
        )
    )(
        temperatures,
        pressures_bar,
        gas_log_amounts,
        gas_total_log_amounts,
    )

    def condensate_layer(
        temperature,
        pressure,
        inventory,
        q_init,
        qtot_init,
        graphite_init,
    ):
        return minimize_gibbs_fixed_support_with_diagnostics(
            ThermoState(
                temperature,
                jnp.log(pressure / REFERENCE_PRESSURE_BAR),
                inventory,
            ),
            q_init,
            graphite_init.reshape((1,)),
            qtot_init,
            setup.formula_matrix,
            formula_matrix_cond,
            setup.gas_setup.hvector_func,
            graphite_hvector,
            residual_crit=CONDENSATE_RESIDUAL_TOLERANCE,
            max_iter=CONDENSATE_MAX_ITERATIONS,
        )

    fixed_q_init = (
        gas_log_amounts
        if fixed_gas_log_amounts_init is None
        else jnp.asarray(fixed_gas_log_amounts_init)
    )
    fixed_qtot_init = (
        gas_total_log_amounts
        if fixed_total_log_amounts_init is None
        else jnp.asarray(fixed_total_log_amounts_init)
    )
    fixed_result, fixed_diagnostics = jax.vmap(condensate_layer)(
        temperatures,
        pressures_bar,
        element_vectors,
        fixed_q_init,
        fixed_qtot_init,
        graphite_amounts_init,
    )
    return (
        gas_log_amounts,
        gas_diagnostics,
        margins,
        fixed_result,
        fixed_diagnostics,
    )


def prepare_graphite_profile(
    pressures_bar: Any,
    *,
    setup: Optional[Any] = None,
) -> GraphiteProfilePlan:
    """Discover and certify the nominal static graphite layer partition."""

    pressure = jnp.asarray(pressures_bar, dtype=jnp.float64)
    if pressure.ndim != 1 or pressure.shape[0] < 2:
        raise ValueError("pressures_bar must be a one-dimensional profile.")
    if not bool(jnp.all(jnp.isfinite(pressure) & (pressure > 0.0))):
        raise ValueError("pressures_bar must contain finite positive values.")
    if not bool(jnp.all(jnp.diff(pressure) > 0.0)):
        raise ValueError("pressures_bar must be ordered from top to bottom.")

    chemistry = graphite_only_chemical_setup(setup)
    graphite_index = chemistry.condensate_species.index(GRAPHITE_SPECIES)
    carbon_index = chemistry.elements.index("C")
    oxygen_index = chemistry.elements.index("O")
    co_species_index = chemistry.gas_species.index("C1O1")
    element_vector = carbon_rich_reference_vector(chemistry)
    nominal_temperature = powerlaw_temperature(
        pressure, TRUTH_T0_K, TRUTH_ALPHA
    )

    gas_result, gas_diagnostics = solve_gas_profile(
        chemistry.gas_setup,
        nominal_temperature,
        pressure,
        element_vector,
        Pref=REFERENCE_PRESSURE_BAR,
        options=EquilibriumOptions(
            epsilon_crit=GAS_RESIDUAL_TOLERANCE,
            max_iter=GAS_MAX_ITERATIONS,
            method="vmap_cold",
        ),
        return_diagnostics=True,
    )
    if not bool(jnp.all(gas_diagnostics["converged"])):
        failed = np.flatnonzero(
            ~np.asarray(jax.device_get(gas_diagnostics["converged"]), dtype=bool)
        )
        raise RuntimeError(
            "Nominal gas-only preparation did not converge in layers "
            f"{failed.tolist()}."
        )
    gas_total_log_amounts = jnp.log(gas_result.ntot)
    seed_amount = min(
        1.0e-3,
        GRAPHITE_SEED_FRACTION * float(element_vector[carbon_index]),
    )
    graphite_seed = jnp.full(
        (pressure.shape[0],), seed_amount, dtype=element_vector.dtype
    )
    element_vectors = jnp.broadcast_to(
        element_vector, (pressure.shape[0], element_vector.shape[0])
    )
    (
        gas_log_amounts,
        audited_gas_diagnostics,
        margins,
        fixed_result,
        fixed_diagnostics,
    ) = _audit_graphite_candidate(
        chemistry,
        graphite_index,
        nominal_temperature,
        pressure,
        element_vectors,
        gas_result.ln_n,
        gas_total_log_amounts,
        graphite_seed,
    )
    gas_converged = jnp.asarray(audited_gas_diagnostics["converged"])
    fixed_amounts = fixed_result.condensate_amounts[:, 0]
    fixed_valid = (
        fixed_diagnostics.converged
        & jnp.isfinite(fixed_diagnostics.residual_norm)
        & jnp.isfinite(fixed_amounts)
        & (fixed_amounts > MIN_ACTIVE_GRAPHITE_AMOUNT)
    )
    activity_candidate = margins < 0.0
    unresolved = activity_candidate & (~fixed_valid)
    if not bool(jnp.all(gas_converged)) or bool(jnp.any(unresolved)):
        failed = np.flatnonzero(
            np.asarray(jax.device_get((~gas_converged) | unresolved), dtype=bool)
        )
        raise RuntimeError(
            "Nominal graphite support preparation failed in layers "
            f"{failed.tolist()}."
        )
    active_mask = np.asarray(
        jax.device_get(activity_candidate & fixed_valid), dtype=bool
    )
    active_indices = tuple(int(index) for index in np.flatnonzero(active_mask))
    inactive_indices = tuple(int(index) for index in np.flatnonzero(~active_mask))
    if not active_indices or not inactive_indices:
        raise RuntimeError(
            "The demo requires both graphite-active and gas-only layers; "
            f"found active={active_indices}, inactive={inactive_indices}."
        )

    hybrid_q = jnp.where(
        jnp.asarray(active_mask)[:, None],
        fixed_result.gas_log_amounts,
        gas_log_amounts,
    )
    hybrid_qtot = jax.scipy.special.logsumexp(hybrid_q, axis=1)
    graphite_init = jnp.where(
        jnp.asarray(active_mask), fixed_amounts, graphite_seed
    )
    return GraphiteProfilePlan(
        setup=chemistry,
        pressures_bar=pressure,
        nominal_temperatures=nominal_temperature,
        reference_element_vector=element_vector,
        carbon_index=carbon_index,
        oxygen_index=oxygen_index,
        co_species_index=co_species_index,
        graphite_species_index=graphite_index,
        active_indices=active_indices,
        inactive_indices=inactive_indices,
        gas_only_log_amounts=gas_log_amounts,
        gas_only_total_log_amounts=jax.scipy.special.logsumexp(
            gas_log_amounts, axis=1
        ),
        hybrid_log_amounts_init=hybrid_q,
        hybrid_total_log_amounts_init=hybrid_qtot,
        graphite_amounts_init=graphite_init,
        graphite_seed_amount=seed_amount,
        nominal_graphite_driving_margin=margins,
        nominal_fixed_support_residual=fixed_diagnostics.residual_norm,
    )


def solve_hybrid_log_amounts(
    plan: GraphiteProfilePlan,
    temperatures: Any,
    log_co_scale: Any,
) -> jax.Array:
    """Solve the frozen active partition with condensate and gas custom VJPs."""

    temperature = jnp.asarray(temperatures)
    if temperature.shape != plan.pressures_bar.shape:
        raise ValueError("temperatures must have one value per prepared layer.")
    inventory = scale_carbon_and_oxygen(
        plan.reference_element_vector,
        plan.carbon_index,
        plan.oxygen_index,
        log_co_scale,
    )
    _, formula_matrix_cond, graphite_hvector = _graphite_support_arrays(
        plan.setup, plan.graphite_species_index
    )
    result = jnp.zeros_like(plan.hybrid_log_amounts_init)

    active = jnp.asarray(plan.active_indices, dtype=jnp.int32)

    def active_layer(temp, pressure, q_init, qtot_init, graphite_init):
        solved = minimize_gibbs_fixed_support(
            ThermoState(
                temp,
                jnp.log(pressure / REFERENCE_PRESSURE_BAR),
                inventory,
            ),
            q_init,
            graphite_init.reshape((1,)),
            qtot_init,
            plan.setup.formula_matrix,
            formula_matrix_cond,
            plan.setup.gas_setup.hvector_func,
            graphite_hvector,
            residual_crit=CONDENSATE_RESIDUAL_TOLERANCE,
            max_iter=CONDENSATE_MAX_ITERATIONS,
        )
        return solved.gas_log_amounts

    active_q = jax.vmap(active_layer)(
        temperature[active],
        plan.pressures_bar[active],
        plan.hybrid_log_amounts_init[active],
        plan.hybrid_total_log_amounts_init[active],
        plan.graphite_amounts_init[active],
    )
    result = result.at[active].set(active_q)

    inactive = jnp.asarray(plan.inactive_indices, dtype=jnp.int32)

    def inactive_layer(temp, pressure, q_init, qtot_init):
        return minimize_gibbs(
            ThermoState(
                temp,
                jnp.log(pressure / REFERENCE_PRESSURE_BAR),
                inventory,
            ),
            q_init,
            qtot_init,
            plan.setup.formula_matrix,
            plan.setup.gas_setup.hvector_func,
            epsilon_crit=GAS_RESIDUAL_TOLERANCE,
            max_iter=GAS_MAX_ITERATIONS,
        )

    inactive_q = jax.vmap(inactive_layer)(
        temperature[inactive],
        plan.pressures_bar[inactive],
        plan.gas_only_log_amounts[inactive],
        plan.gas_only_total_log_amounts[inactive],
    )
    return result.at[inactive].set(inactive_q)


def co_vmr_profile(
    plan: GraphiteProfilePlan,
    temperatures: Any,
    log_co_scale: Any,
) -> jax.Array:
    """Return the CO volume-mixing-ratio profile from the hybrid solver."""

    q = solve_hybrid_log_amounts(plan, temperatures, log_co_scale)
    log_total = jax.scipy.special.logsumexp(q, axis=1)
    return jnp.exp(q[:, plan.co_species_index] - log_total)


def _prior_corners() -> tuple[tuple[float, float, float], ...]:
    return tuple(
        itertools.product(
            T0_PRIOR_BOUNDS_K,
            ALPHA_PRIOR_BOUNDS,
            LOG_CO_SCALE_PRIOR_BOUNDS,
        )
    )


def preflight_graphite_plan(plan: GraphiteProfilePlan) -> dict[str, Any]:
    """Check support identity, primal convergence, and reverse-mode gradients."""

    nominal_mask = plan.active_mask
    condensate_metadata = plan.setup.condensate_setup.metadata or {}
    graphite_validity = plan.setup.condensate_setup.temperature_validity_upper
    graphite_temperature_upper = (
        float(graphite_validity[0])
        if graphite_validity is not None
        else float("inf")
    )
    corner_rows: list[dict[str, Any]] = []
    for t0_kelvin, alpha, log_co_scale in _prior_corners():
        temperature = powerlaw_temperature(
            plan.pressures_bar, t0_kelvin, alpha
        )
        inventory = scale_carbon_and_oxygen(
            plan.reference_element_vector,
            plan.carbon_index,
            plan.oxygen_index,
            log_co_scale,
        )
        inventories = jnp.broadcast_to(
            inventory,
            (
                plan.pressures_bar.shape[0],
                plan.reference_element_vector.shape[0],
            ),
        )
        (
            _gas_q,
            gas_diagnostics,
            margins,
            fixed_result,
            fixed_diagnostics,
        ) = _audit_graphite_candidate(
            plan.setup,
            plan.graphite_species_index,
            temperature,
            plan.pressures_bar,
            inventories,
            plan.gas_only_log_amounts,
            plan.gas_only_total_log_amounts,
            plan.graphite_amounts_init,
            fixed_gas_log_amounts_init=plan.hybrid_log_amounts_init,
            fixed_total_log_amounts_init=plan.hybrid_total_log_amounts_init,
        )
        amounts = fixed_result.condensate_amounts[:, 0]
        valid_fixed = (
            fixed_diagnostics.converged
            & jnp.isfinite(fixed_diagnostics.residual_norm)
            & jnp.isfinite(amounts)
            & (amounts > MIN_ACTIVE_GRAPHITE_AMOUNT)
        )
        corner_mask = np.asarray(
            jax.device_get((margins < 0.0) & valid_fixed), dtype=bool
        )
        gas_ok = bool(jnp.all(gas_diagnostics["converged"]))
        mask_ok = bool(np.array_equal(corner_mask, nominal_mask))
        valid_fixed_host = np.asarray(jax.device_get(valid_fixed), dtype=bool)
        active_fixed_ok = bool(np.all(valid_fixed_host[nominal_mask]))
        fixed_residuals = np.asarray(
            jax.device_get(fixed_diagnostics.residual_norm), dtype=float
        )
        active_residual_max = float(np.max(fixed_residuals[nominal_mask]))
        active_amount_min = float(
            np.min(np.asarray(jax.device_get(amounts))[nominal_mask])
        )
        inactive_margin_min = float(
            np.min(np.asarray(jax.device_get(margins))[~nominal_mask])
        )
        gas_budget = jnp.einsum(
            "ek,lk->le",
            plan.setup.formula_matrix,
            jnp.exp(fixed_result.gas_log_amounts),
        )
        condensate_budget = (
            fixed_result.condensate_amounts @ plan.setup.formula_matrix_cond.T
        )
        budget_residual = gas_budget + condensate_budget - inventories
        budget_scale = jnp.where(
            jnp.abs(inventories) > 0.0,
            jnp.abs(inventories),
            jnp.ones_like(inventories),
        )
        scaled_budget_residual = jnp.max(
            jnp.abs(budget_residual) / budget_scale,
            axis=1,
        )
        active_budget_residual_max = float(
            np.max(
                np.asarray(jax.device_get(scaled_budget_residual))[nominal_mask]
            )
        )
        temperature_valid = bool(
            jnp.all(temperature <= graphite_temperature_upper)
        )
        corner_ok = (
            gas_ok
            and mask_ok
            and active_fixed_ok
            and active_residual_max <= CONDENSATE_RESIDUAL_TOLERANCE
            and active_budget_residual_max <= CONDENSATE_RESIDUAL_TOLERANCE
            and active_amount_min > MIN_ACTIVE_GRAPHITE_AMOUNT
            and inactive_margin_min > MIN_INACTIVE_DRIVING_MARGIN
            and temperature_valid
        )
        corner_rows.append(
            {
                "t0_kelvin": t0_kelvin,
                "alpha": alpha,
                "log_co_scale": log_co_scale,
                "active_indices": np.flatnonzero(corner_mask).tolist(),
                "gas_converged": gas_ok,
                "active_fixed_support_converged": active_fixed_ok,
                "support_matches_nominal": mask_ok,
                "maximum_active_fixed_support_residual": active_residual_max,
                "maximum_active_scaled_budget_residual": (
                    active_budget_residual_max
                ),
                "minimum_active_graphite_amount": active_amount_min,
                "minimum_inactive_driving_margin": inactive_margin_min,
                "graphite_temperature_valid": temperature_valid,
                "uses_frozen_nuts_initialization": True,
                "passed": corner_ok,
            }
        )

    def chemistry_summary(t0_kelvin, alpha, log_co_scale):
        temperature = powerlaw_temperature(
            plan.pressures_bar, t0_kelvin, alpha
        )
        vmr = co_vmr_profile(plan, temperature, log_co_scale)
        return jnp.sum(jnp.log(jnp.clip(vmr, 1.0e-300)))

    value, gradient = jax.value_and_grad(
        chemistry_summary, argnums=(0, 1, 2)
    )(
        jnp.asarray(TRUTH_T0_K, dtype=jnp.float64),
        jnp.asarray(TRUTH_ALPHA, dtype=jnp.float64),
        jnp.asarray(TRUTH_LOG_CO_SCALE, dtype=jnp.float64),
    )
    gradient_values = tuple(float(item) for item in gradient)
    gradient_finite = bool(
        np.all(np.isfinite(np.asarray(gradient_values, dtype=float)))
    )
    report = {
        "schema": "exogibbs_condensate_fixed_support_nuts_preflight_v1",
        "case_name": CASE_NAME,
        "local_fixed_support_contract": True,
        "condensate_model_scope": CONDENSATE_MODEL_SCOPE,
        "condensate_catalog_mode": "reduced_explicit",
        "reduced_condensate_catalog": True,
        "condensate_catalog_species": [GRAPHITE_SPECIES],
        "full_fastchem4_condensate_species_count": condensate_metadata.get(
            "full_fastchem4_condensate_species_count"
        ),
        "excluded_condensate_species_count": condensate_metadata.get(
            "excluded_condensate_species_count"
        ),
        "full_catalog_equilibrium_claimed": False,
        "full_catalog_support_closure_checked": False,
        "full_fastchem4_catalog_support_closure_checked": False,
        "differentiates_support_discovery": False,
        "differentiates_rainout": False,
        "rainout": False,
        "carbon_to_oxygen_ratio": CARBON_TO_OXYGEN_RATIO,
        "graphite_species": GRAPHITE_SPECIES,
        "layer_count": int(plan.pressures_bar.shape[0]),
        "active_indices": list(plan.active_indices),
        "inactive_indices": list(plan.inactive_indices),
        "prior_bounds": {
            "t0_kelvin": list(T0_PRIOR_BOUNDS_K),
            "alpha": list(ALPHA_PRIOR_BOUNDS),
            "log_co_scale": list(LOG_CO_SCALE_PRIOR_BOUNDS),
        },
        "corner_checks": corner_rows,
        "chemistry_log_co_vmr_sum": float(value),
        "chemistry_reverse_mode_gradient": {
            "t0_kelvin": gradient_values[0],
            "alpha": gradient_values[1],
            "log_co_scale": gradient_values[2],
        },
        "gradient_finite": gradient_finite,
    }
    report["passed"] = bool(
        gradient_finite and all(row["passed"] for row in corner_rows)
    )
    if not report["passed"]:
        failed = [row for row in corner_rows if not row["passed"]]
        raise RuntimeError(
            "Fixed-support preflight failed.  The requested layer grid or "
            "prior box is not local to one graphite support. Failed corners: "
            f"{failed}"
        )
    return report


def build_condensate_model(
    context: Any,
    plan: GraphiteProfilePlan,
    observation: Any,
):
    """Build the NumPyro model without importing NumPyro at module import."""

    import numpyro
    import numpyro.distributions as dist

    def model():
        t0_kelvin = numpyro.sample(
            "T0",
            dist.Uniform(*T0_PRIOR_BOUNDS_K),
        )
        alpha = numpyro.sample(
            "alpha",
            dist.Uniform(*ALPHA_PRIOR_BOUNDS),
        )
        log_co_scale = numpyro.sample(
            "log_co_scale",
            dist.Uniform(*LOG_CO_SCALE_PRIOR_BOUNDS),
        )
        temperature = powerlaw_temperature(
            plan.pressures_bar, t0_kelvin, alpha
        )
        co_vmr = co_vmr_profile(plan, temperature, log_co_scale)
        raw_flux = co_emission_flux(context, temperature, co_vmr)
        normalized_flux = raw_flux / observation.flux_scale
        numpyro.sample(
            "spectrum",
            dist.Normal(normalized_flux, observation.noise_std),
            obs=observation.observed,
        )

    return model


def _json_ready_plan_metadata(plan: GraphiteProfilePlan) -> dict[str, Any]:
    condensate_metadata = plan.setup.condensate_setup.metadata or {}
    return {
        "condensate_model_scope": CONDENSATE_MODEL_SCOPE,
        "condensate_catalog_mode": "reduced_explicit",
        "reduced_condensate_catalog": True,
        "support_species": GRAPHITE_SPECIES,
        "active_indices": list(plan.active_indices),
        "inactive_indices": list(plan.inactive_indices),
        "pressures_bar": np.asarray(plan.pressures_bar, dtype=float).tolist(),
        "nominal_temperatures_kelvin": np.asarray(
            plan.nominal_temperatures, dtype=float
        ).tolist(),
        "nominal_graphite_inactive_margin": np.asarray(
            plan.nominal_graphite_driving_margin, dtype=float
        ).tolist(),
        "nominal_fixed_support_residual": np.asarray(
            plan.nominal_fixed_support_residual, dtype=float
        ).tolist(),
        "graphite_seed_amount": plan.graphite_seed_amount,
        "gas_species_count": len(plan.setup.gas_species),
        "condensate_catalog_count": len(plan.setup.condensate_species),
        "source_condensate_catalog_count": condensate_metadata.get(
            "source_condensate_catalog_count"
        ),
        "excluded_condensate_species_count": condensate_metadata.get(
            "excluded_condensate_species_count"
        ),
        "full_catalog_equilibrium_claimed": False,
        "full_catalog_support_closure_checked": False,
        "element_count": len(plan.setup.elements),
    }


def _preflight_output_path(output_directory: Union[str, Path]) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    return output / f"{CASE_NAME}_preflight.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser shared with the gas retrieval demos."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_arguments(
        parser,
        default_output_dir=Path("results") / "vjp_retrieval" / CASE_NAME,
    )
    # Eight layers leave a useful gap between the last graphite layer and the
    # first gas-only layer.  Other grids are allowed only when preflight passes.
    parser.set_defaults(nlayer=8)
    parser.add_argument(
        "--relative-noise",
        type=float,
        default=DEFAULT_RELATIVE_NOISE,
        help="Gaussian mock-noise standard deviation relative to flux scale.",
    )
    return parser


def _print_preflight_summary(report: Mapping[str, Any], path: Path) -> None:
    print("Condensate fixed-support preflight passed.")
    print(f"  graphite-active layers: {report['active_indices']}")
    print(f"  gas-only layers: {report['inactive_indices']}")
    print(f"  report: {path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run chemistry preflight and, unless requested otherwise, NUTS."""

    parser = build_parser()
    args = parser.parse_args(argv)
    settings = resolve_run_settings(args)
    nlayer, nu_points = resolve_demo_shape(args)

    context = None
    if not args.preflight_only or args.co_database is not None:
        context = build_spectral_context(
            args.co_database,
            nlayer=nlayer,
            nu_points=nu_points,
        )
        pressures = jnp.asarray(context.art.pressure, dtype=jnp.float64)
    else:
        pressures = pressure_profile(nlayer)

    plan = prepare_graphite_profile(pressures)
    preflight = preflight_graphite_plan(plan)
    preflight["plan"] = _json_ready_plan_metadata(plan)
    if args.relative_noise <= 0.0:
        raise ValueError("--relative-noise must be positive.")
    if context is None:
        preflight_path = _preflight_output_path(args.output_dir)
        preflight_path.write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _print_preflight_summary(preflight, preflight_path)
        return

    truth_temperature = powerlaw_temperature(
        plan.pressures_bar, TRUTH_T0_K, TRUTH_ALPHA
    )
    truth_co_vmr = co_vmr_profile(
        plan, truth_temperature, TRUTH_LOG_CO_SCALE
    )
    truth_flux = co_emission_flux(
        context, truth_temperature, truth_co_vmr
    )
    observation = make_mock_observation(
        truth_flux,
        seed=settings.seed,
        relative_noise=args.relative_noise,
    )

    def spectral_loss(t0_kelvin, alpha, log_co_scale):
        temperature = powerlaw_temperature(
            plan.pressures_bar, t0_kelvin, alpha
        )
        co_vmr = co_vmr_profile(plan, temperature, log_co_scale)
        prediction = co_emission_flux(context, temperature, co_vmr)
        prediction = prediction / observation.flux_scale
        return jnp.mean(jnp.square(prediction - observation.observed))

    spectral_loss_value, spectral_gradient = jax.value_and_grad(
        spectral_loss, argnums=(0, 1, 2)
    )(
        jnp.asarray(TRUTH_T0_K, dtype=jnp.float64),
        jnp.asarray(TRUTH_ALPHA, dtype=jnp.float64),
        jnp.asarray(TRUTH_LOG_CO_SCALE, dtype=jnp.float64),
    )
    spectral_gradient_values = tuple(float(value) for value in spectral_gradient)
    if not np.isfinite(float(spectral_loss_value)) or not np.all(
        np.isfinite(np.asarray(spectral_gradient_values))
    ):
        raise RuntimeError(
            "Reverse-mode ExoJAX spectrum preflight produced non-finite values."
        )
    preflight["spectral_loss"] = float(spectral_loss_value)
    preflight["spectral_reverse_mode_gradient"] = {
        "t0_kelvin": spectral_gradient_values[0],
        "alpha": spectral_gradient_values[1],
        "log_co_scale": spectral_gradient_values[2],
    }
    preflight_path = _preflight_output_path(args.output_dir)
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_preflight_summary(preflight, preflight_path)
    if args.preflight_only:
        write_run_outputs(
            args.output_dir,
            case_name=CASE_NAME,
            context=context,
            observation=observation,
            metadata={"preflight": preflight},
        )
        return

    model = build_condensate_model(context, plan, observation)
    mcmc = run_reverse_mode_nuts(model, observation, settings)
    write_run_outputs(
        args.output_dir,
        case_name=CASE_NAME,
        context=context,
        observation=observation,
        mcmc=mcmc,
        metadata={
            "preflight": preflight,
            "contract": (
                "Reduced FastChem4-gas plus C(s)-only condensate model. Local "
                "fixed support only; full-catalog phase closure, support "
                "discovery, phase transitions, production lifecycle, and "
                "rainout are outside AD."
            ),
        },
    )
    print(f"{CASE_NAME}: completed; outputs: {args.output_dir}")


if __name__ == "__main__":
    main()

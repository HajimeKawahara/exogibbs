"""Default-off FastChem4 public condensate output adapter.

This module is intentionally not imported from :mod:`exogibbs.presets`.
It packages the FC4-M021/M022 public-output contract for diagnostic use
without changing existing FastChem presets, solver paths, or defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence


EXPECTED_FASTCHEM4_COMMIT = "0e1b620aa859530ae6fd5413907794c544a9f376"
EXPECTED_FASTCHEM4_TAG = "v4.0.1-1-g0e1b620"
PUBLIC_BUDGET_REL_TOLERANCE = 1.0e-6


def normalise_element_abundances(raw_abundances: Sequence[float]) -> tuple[float, ...]:
    """Normalise FastChem4 public H=1 element abundances to epsilon."""

    values = tuple(float(value) for value in raw_abundances)
    total = sum(values)
    if total == 0.0:
        return tuple(0.0 for _ in values)
    return tuple(value / total for value in values)


@dataclass(frozen=True)
class FastChem4PublicCondensateOutputAdapter:
    """Explicit opt-in adapter for FastChem4 public condensate outputs."""

    element_order: tuple[str, ...]
    gas_species_order: tuple[str, ...]
    condensate_species_order: tuple[str, ...]
    raw_element_abundances: tuple[float, ...]
    gas_stoichiometry: tuple[tuple[float, ...], ...]
    condensate_stoichiometry: tuple[tuple[float, ...], ...]
    fastchem_runtime: Any | None = field(default=None, repr=False, compare=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "provider_name": "FastChem4PublicCondensateOutputAdapter",
            "default_off": True,
            "diagnostic_only": True,
            "production_defaults_changed": False,
            "public_only": True,
            "selected_fastchem4_commit": EXPECTED_FASTCHEM4_COMMIT,
            "selected_fastchem4_tag": EXPECTED_FASTCHEM4_TAG,
            "branch_exact_residual_status": "not_computed_public_api_missing_internal_state",
            "positive_support_semantics": "final positive condensate output support; not internal active set or stable set",
            "normalised_element_epsilon_rule": "normalise FastChem.getElementAbundances() before public budget comparisons",
            "forbidden_constructor_inputs": (
                "temporary snapshot trace values",
                "internal active condensate set",
                "internal stable condensate set",
                "branch-exact J matrix",
                "branch-exact RHS vector",
            ),
        }

    def normalised_element_epsilon(self) -> tuple[float, ...]:
        return normalise_element_abundances(self.raw_element_abundances)

    def schema(self) -> dict[str, Any]:
        return {
            "element_order": self.element_order,
            "gas_species_count": len(self.gas_species_order),
            "condensate_species_order": self.condensate_species_order,
            "raw_element_abundance_sum": sum(self.raw_element_abundances),
            "normalised_element_epsilon_sum": sum(self.normalised_element_epsilon()),
        }

    def evaluate_public_output(
        self,
        *,
        gas_number_densities: Sequence[float],
        condensate_number_densities: Sequence[float],
        element_cond_degree: Sequence[float],
        total_element_density: float,
        convergence_flag: int | None = None,
        iteration_count: int | None = None,
        condensation_iteration_count: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate public condensate output diagnostics without trace-only fields."""

        gas = tuple(float(value) for value in gas_number_densities)
        condensates = tuple(float(value) for value in condensate_number_densities)
        degrees = tuple(float(value) for value in element_cond_degree)
        total_density = float(total_element_density)
        gas_inventory = self._inventory(gas, self.gas_stoichiometry)
        condensate_inventory = self._inventory(condensates, self.condensate_stoichiometry)
        combined_inventory = tuple(
            gas_value + cond_value
            for gas_value, cond_value in zip(gas_inventory, condensate_inventory)
        )
        epsilon = self.normalised_element_epsilon()
        positive = self._positive_condensate_output_support(condensates)
        budget_rows = []
        degree_rows = []
        for index, symbol in enumerate(self.element_order):
            target = total_density * epsilon[index]
            budget_rel = abs(combined_inventory[index] - target) / (abs(target) + 1.0e-300)
            denom = combined_inventory[index]
            degree_from_inventory = 0.0 if denom == 0.0 else condensate_inventory[index] / denom
            budget_rows.append(
                {
                    "element": symbol,
                    "combined_inventory": combined_inventory[index],
                    "normalised_budget_target": target,
                    "relative_delta": budget_rel,
                }
            )
            degree_rows.append(
                {
                    "element": symbol,
                    "public_degree": degrees[index],
                    "inventory_degree": degree_from_inventory,
                    "absolute_delta": abs(degrees[index] - degree_from_inventory),
                }
            )
        budget_rows.sort(key=lambda row: row["relative_delta"], reverse=True)
        degree_rows.sort(key=lambda row: row["absolute_delta"], reverse=True)
        nonfinite = self._nonfinite_outputs(
            [
                ("gas_number_densities", gas),
                ("condensate_number_densities", condensates),
                ("element_cond_degree", degrees),
            ]
        )
        return {
            "convergence_flag": convergence_flag,
            "iteration_count": iteration_count,
            "condensation_iteration_count": condensation_iteration_count,
            "total_element_density": total_density,
            "shape_checks": {
                "element_count": len(self.element_order),
                "gas_species_count": len(self.gas_species_order),
                "condensate_species_count": len(self.condensate_species_order),
                "gas_density_length": len(gas),
                "condensate_density_length": len(condensates),
                "element_cond_degree_length": len(degrees),
                "all_lengths_match": len(gas) == len(self.gas_species_order)
                and len(condensates) == len(self.condensate_species_order)
                and len(degrees) == len(self.element_order),
            },
            "finite_check": {
                "nonfinite_count": len(nonfinite),
                "nonfinite_outputs": nonfinite,
            },
            "positive_condensate_output_support": {
                "semantics": "final positive condensate output support; not internal active set or stable set",
                "positive_count": len(positive),
                "top_positive_condensates": positive[:12],
            },
            "element_cond_degree_consistency": {
                "max_abs_delta": degree_rows[0]["absolute_delta"] if degree_rows else None,
                "top_delta_rows": degree_rows[:10],
                "status": "passed"
                if degree_rows and degree_rows[0]["absolute_delta"] < PUBLIC_BUDGET_REL_TOLERANCE
                else "failed",
            },
            "normalised_budget_closure": {
                "max_relative_delta": budget_rows[0]["relative_delta"] if budget_rows else None,
                "relative_tolerance": PUBLIC_BUDGET_REL_TOLERANCE,
                "top_relative_delta_rows": budget_rows[:10],
                "status": "passed"
                if budget_rows and budget_rows[0]["relative_delta"] < PUBLIC_BUDGET_REL_TOLERANCE
                else "failed",
            },
            "branch_exact_residual_status": "not_computed_public_api_missing_internal_state",
        }

    def evaluate_case(
        self,
        temperature: float,
        pressure: float,
        *,
        equilibrium_condensation: bool = True,
        rainout_condensation: bool = False,
    ) -> dict[str, Any]:
        """Run a bounded public FastChem4 case when an explicit runtime is attached."""

        if self.fastchem_runtime is None:
            raise RuntimeError("FastChem4 public adapter has no attached fastchem_runtime")
        import pyfastchem  # type: ignore[import-not-found]

        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()
        input_data.temperature = [float(temperature)]
        input_data.pressure = [float(pressure)]
        input_data.equilibrium_condensation = bool(equilibrium_condensation)
        input_data.rainout_condensation = bool(rainout_condensation)
        flag = int(self.fastchem_runtime.calcDensities(input_data, output_data))
        result = self.evaluate_public_output(
            gas_number_densities=output_data.number_densities[0],
            condensate_number_densities=output_data.number_densities_cond[0],
            element_cond_degree=output_data.element_cond_degree[0],
            total_element_density=output_data.total_element_density[0],
            convergence_flag=flag,
            iteration_count=int(output_data.nb_iterations[0]) if output_data.nb_iterations else None,
            condensation_iteration_count=int(output_data.nb_cond_iterations[0])
            if output_data.nb_cond_iterations
            else None,
        )
        result.update(
            {
                "case_id": f"T{float(temperature):g}_P{float(pressure):g}",
                "temperature": float(temperature),
                "pressure": float(pressure),
                "message": pyfastchem.FASTCHEM_MSG[flag],
            }
        )
        return result

    def _inventory(
        self,
        densities: Sequence[float],
        stoichiometry: Sequence[Sequence[float]],
    ) -> tuple[float, ...]:
        out = [0.0 for _ in self.element_order]
        for species_index, density in enumerate(densities):
            for element_index, coeff in enumerate(stoichiometry[species_index]):
                if coeff:
                    out[element_index] += float(coeff) * float(density)
        return tuple(out)

    def _positive_condensate_output_support(
        self,
        condensate_number_densities: Sequence[float],
    ) -> list[dict[str, Any]]:
        positive = [
            {"species": self.condensate_species_order[index], "index": index, "density": float(value)}
            for index, value in enumerate(condensate_number_densities)
            if math.isfinite(float(value)) and float(value) > 0.0
        ]
        positive.sort(key=lambda row: row["density"], reverse=True)
        return positive

    @staticmethod
    def _nonfinite_outputs(named_values: Sequence[tuple[str, Sequence[float]]]) -> list[dict[str, Any]]:
        out = []
        for label, values in named_values:
            for index, value in enumerate(values):
                if not math.isfinite(float(value)):
                    out.append({"array": label, "index": index})
        return out


def build_fastchem4_public_condensate_output_adapter(
    fastchem_runtime: Any | None = None,
    fastchem4_root: str | Path | None = None,
) -> FastChem4PublicCondensateOutputAdapter:
    """Build the explicit default-off public condensate output adapter."""

    runtime = fastchem_runtime
    if runtime is None:
        import pyfastchem  # type: ignore[import-not-found]

        root = Path(fastchem4_root) if fastchem4_root is not None else _repo_root() / "FastChem4"
        runtime = pyfastchem.FastChem(
            str(root / "input/element_abundances/asplund_2021.dat"),
            str(root / "input/logK/logK_wo_ions.dat"),
            str(root / "input/logK/logK_condensates.dat"),
            0,
        )
    element_order = tuple(runtime.getElementSymbol(index) for index in range(runtime.getElementNumber()))
    gas_species_order = tuple(
        runtime.getGasSpeciesSymbol(index) for index in range(runtime.getGasSpeciesNumber())
    )
    condensate_species_order = tuple(
        runtime.getCondSpeciesSymbol(index) for index in range(runtime.getCondSpeciesNumber())
    )
    gas_stoichiometry = tuple(
        tuple(float(value) for value in runtime.getGasSpeciesStoichiometry(index))
        for index in range(runtime.getGasSpeciesNumber())
    )
    condensate_stoichiometry = tuple(
        tuple(float(value) for value in runtime.getCondSpeciesStoichiometry(index))
        for index in range(runtime.getCondSpeciesNumber())
    )
    raw_abundances = tuple(float(value) for value in runtime.getElementAbundances())
    return FastChem4PublicCondensateOutputAdapter(
        element_order=element_order,
        gas_species_order=gas_species_order,
        condensate_species_order=condensate_species_order,
        raw_element_abundances=raw_abundances,
        gas_stoichiometry=gas_stoichiometry,
        condensate_stoichiometry=condensate_stoichiometry,
        fastchem_runtime=runtime,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

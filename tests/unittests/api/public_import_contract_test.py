"""Fresh-process contracts for canonical and compatibility API imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
GENERIC_MAGMA_GAS_EXPORTS = {
    "MagmaGasConditions",
    "MagmaGasDiagnostics",
    "MagmaGasEquilibriumState",
    "MagmaGasInit",
    "MagmaGasModel",
    "MagmaGasModelEvaluation",
    "MagmaGasOptions",
    "MagmaGasProblem",
    "MagmaGasResult",
    "solve",
}
MELTYQ_PRESET_EXPORTS = {
    "MELTYQ_ELEMENTS",
    "MELTYQ_MEAN_MELT_MOLAR_MASS_G_MOL",
    "MELTYQ_MELT_QUANTITIES",
    "MELTYQ_ROOT_RESIDUALS",
    "MELTYQ_SPECIES",
    "MeltyqMagmaGasInputs",
    "MeltyqMagmaGasState",
    "prepare_meltyq_problem",
}


def _run_import_script(script: str) -> dict:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])


def test_base_api_import_is_lazy_and_exports_are_deterministic() -> None:
    payload = _run_import_script(
        """
        import json
        import sys
        import types

        import exogibbs.api as api

        print(json.dumps({
            "condensate_loaded": (
                "exogibbs.api.condensate_equilibrium" in sys.modules
            ),
            "gas_loaded": "exogibbs.api.equilibrium" in sys.modules,
            "magma_gas_api_loaded": (
                "exogibbs.api.magma_gas" in sys.modules
            ),
            "applications_loaded": any(
                name == "exogibbs.applications"
                or name.startswith("exogibbs.applications.")
                for name in sys.modules
            ),
            "magma_gas_implementation_loaded": any(
                name == "exogibbs.applications.magma_gas"
                or name.startswith("exogibbs.applications.magma_gas.")
                for name in sys.modules
            ),
            "magma_gas_preset_loaded": (
                "exogibbs.presets.magma_gas" in sys.modules
            ),
            "all_unique": len(api.__all__) == len(set(api.__all__)),
            "child_names": [
                name for name in (
                    "gas",
                    "condensate",
                    "magma_gas",
                    "equilibrium",
                    "condensate_equilibrium",
                )
                if name in api.__all__
            ],
        }))
        """
    )

    assert not payload["condensate_loaded"]
    assert not payload["gas_loaded"]
    assert not payload["magma_gas_api_loaded"]
    assert not payload["applications_loaded"]
    assert not payload["magma_gas_implementation_loaded"]
    assert not payload["magma_gas_preset_loaded"]
    assert payload["all_unique"]
    assert payload["child_names"] == [
        "gas",
        "condensate",
        "magma_gas",
        "equilibrium",
        "condensate_equilibrium",
    ]


def test_canonical_modules_alias_existing_public_callables() -> None:
    payload = _run_import_script(
        """
        import json
        import types

        import exogibbs.api as api
        from exogibbs.api import condensate, gas
        from exogibbs.api.condensate_equilibrium import (
            CondensateEquilibriumOptions,
            condensate_equilibrium,
            condensate_equilibrium_profile,
        )
        from exogibbs.api.equilibrium import (
            EquilibriumOptions,
            equilibrium,
            equilibrium_profile,
        )

        print(json.dumps({
            "gas_module": isinstance(gas, types.ModuleType),
            "condensate_module": isinstance(condensate, types.ModuleType),
            "lnphi_type_alias": (
                api.LogFugacityCoefficientFunction
                is gas.LogFugacityCoefficientFunction
                is condensate.LogFugacityCoefficientFunction
            ),
            "gas_solve_alias": gas.solve is equilibrium,
            "gas_profile_alias": gas.solve_profile is equilibrium_profile,
            "gas_options_alias": gas.EquilibriumOptions is EquilibriumOptions,
            "condensate_solve_alias": (
                condensate.solve is condensate_equilibrium
            ),
            "condensate_profile_alias": (
                condensate.solve_profile is condensate_equilibrium_profile
            ),
            "condensate_options_alias": (
                condensate.CondensateEquilibriumOptions
                is CondensateEquilibriumOptions
            ),
        }))
        """
    )

    assert all(payload.values())


def test_magma_gas_api_exports_only_the_generic_engine() -> None:
    payload = _run_import_script(
        """
        import json
        import sys
        import types

        import exogibbs.api as api
        import exogibbs.applications.magma_gas as implementation
        from exogibbs.api import magma_gas

        type_names = (
            "MagmaGasConditions",
            "MagmaGasDiagnostics",
            "MagmaGasEquilibriumState",
            "MagmaGasInit",
            "MagmaGasModel",
            "MagmaGasModelEvaluation",
            "MagmaGasOptions",
            "MagmaGasProblem",
            "MagmaGasResult",
        )

        print(json.dumps({
            "is_module": isinstance(magma_gas, types.ModuleType),
            "umbrella_identity": api.magma_gas is magma_gas,
            "solve_identity": magma_gas.solve is implementation.solve,
            "type_identities": all(
                getattr(magma_gas, name) is getattr(implementation, name)
                for name in type_names
            ),
            "exports": sorted(magma_gas.__all__),
            "meltyq_exports": sorted(
                name for name in magma_gas.__all__
                if "meltyq" in name.casefold()
            ),
            "has_profile": hasattr(magma_gas, "solve_profile"),
            "has_legacy_solver": hasattr(
                magma_gas,
                "solve_magma_atmosphere_interface",
            ),
            "preset_loaded": (
                "exogibbs.presets.magma_gas" in sys.modules
            ),
            "meltyq_model_loaded": (
                "exogibbs.applications.magma_gas.models.meltyq" in sys.modules
            ),
            "other_application_loaded": any(
                name.startswith("exogibbs.applications.")
                and name != "exogibbs.applications.magma_gas"
                and not name.startswith(
                    "exogibbs.applications.magma_gas."
                )
                for name in sys.modules
            ),
            "experimental_loaded": any(
                name == "exogibbs.experimental"
                or name.startswith("exogibbs.experimental.")
                for name in sys.modules
            ),
        }))
        """
    )

    assert payload["is_module"]
    assert payload["umbrella_identity"]
    assert payload["solve_identity"]
    assert payload["type_identities"]
    assert set(payload["exports"]) == GENERIC_MAGMA_GAS_EXPORTS
    assert payload["meltyq_exports"] == []
    assert not payload["has_profile"]
    assert not payload["has_legacy_solver"]
    assert not payload["preset_loaded"]
    assert not payload["meltyq_model_loaded"]
    assert not payload["other_application_loaded"]
    assert not payload["experimental_loaded"]


def test_meltyq_preset_aliases_builtin_model_without_experimental_import() -> None:
    payload = _run_import_script(
        """
        import importlib
        import json
        import sys

        implementation = importlib.import_module(
            "exogibbs.applications.magma_gas.models.meltyq"
        )
        preset = importlib.import_module("exogibbs.presets.magma_gas")

        print(json.dumps({
            "exports": sorted(preset.__all__),
            "identities": all(
                getattr(preset, name) is getattr(implementation, name)
                for name in preset.__all__
            ),
            "experimental_loaded": any(
                name == "exogibbs.experimental"
                or name.startswith("exogibbs.experimental.")
                for name in sys.modules
            ),
            "api_loaded": any(
                name == "exogibbs.api"
                or name.startswith("exogibbs.api.")
                for name in sys.modules
            ),
        }))
        """
    )

    assert set(payload["exports"]) == MELTYQ_PRESET_EXPORTS
    assert payload["identities"]
    assert not payload["experimental_loaded"]
    assert not payload["api_loaded"]


def test_colliding_umbrella_names_are_modules_for_every_import_order() -> None:
    for preload in ("", "gas", "condensate", "legacy"):
        payload = _run_import_script(
            f"""
            import importlib
            import json
            import types

            preload = {preload!r}
            if preload == "gas":
                importlib.import_module("exogibbs.api.gas")
            elif preload == "condensate":
                importlib.import_module("exogibbs.api.condensate")
            elif preload == "legacy":
                importlib.import_module(
                    "exogibbs.api.condensate_equilibrium"
                )

            from exogibbs.api import condensate_equilibrium, equilibrium

            print(json.dumps({{
                "equilibrium": isinstance(equilibrium, types.ModuleType),
                "condensate_equilibrium": isinstance(
                    condensate_equilibrium,
                    types.ModuleType,
                ),
            }}))
            """
        )

        assert payload == {
            "equilibrium": True,
            "condensate_equilibrium": True,
        }


def test_existing_noncolliding_umbrella_exports_remain_available() -> None:
    payload = _run_import_script(
        """
        import json

        from exogibbs.api import (
            ChemicalSetup,
            CondensateChemicalSetup,
            CondensateEquilibriumOptions,
            CondensateEquilibriumResult,
            EquilibriumGrid,
            EquilibriumGridMetadata,
            EquilibriumInit,
            EquilibriumOptions,
            EquilibriumResult,
            HEAD_ROUTE_V2,
            build_condensate_chemical_setup,
            build_equilibrium_grid,
            condensate_equilibrium_profile,
        )

        print(json.dumps({
            "names": [
                value.__name__
                for value in (
                    ChemicalSetup,
                    CondensateChemicalSetup,
                    CondensateEquilibriumOptions,
                    CondensateEquilibriumResult,
                    EquilibriumGrid,
                    EquilibriumGridMetadata,
                    EquilibriumInit,
                    EquilibriumOptions,
                    EquilibriumResult,
                    build_condensate_chemical_setup,
                    build_equilibrium_grid,
                    condensate_equilibrium_profile,
                )
            ],
            "route": HEAD_ROUTE_V2,
        }))
        """
    )

    assert payload["route"] == "head_v2"
    assert len(payload["names"]) == 12


def test_model_compatibility_exports_preserve_identity() -> None:
    from exogibbs.api.chemistry import (
        ChemicalSetup as CompatibilityChemicalSetup,
    )
    from exogibbs.api.chemistry import ThermoState as CompatibilityThermoState
    from exogibbs.api.condensate_equilibrium import (
        CondensateChemicalSetup as CompatibilityCondensateChemicalSetup,
    )
    from exogibbs.equilibrium.condensate.setup import CondensateChemicalSetup
    from exogibbs.equilibrium.gas.types import ThermoState
    from exogibbs.thermo.models import ChemicalSetup

    assert CompatibilityChemicalSetup is ChemicalSetup
    assert CompatibilityThermoState is ThermoState
    assert CompatibilityCondensateChemicalSetup is CondensateChemicalSetup


def test_internal_compatibility_modules_preserve_identity() -> None:
    import importlib

    old_equations = importlib.import_module("exogibbs.optimize.core")
    new_equations = importlib.import_module(
        "exogibbs.equilibrium.gas.kernel.equations"
    )
    old_solver = importlib.import_module("exogibbs.optimize.minimize")
    new_solver = importlib.import_module(
        "exogibbs.equilibrium.gas.kernel.solver"
    )
    old_potential = importlib.import_module("exogibbs.api.potential")
    new_potential = importlib.import_module("exogibbs.thermo.potential")
    old_grid = importlib.import_module("exogibbs.api.equilibrium_grid")
    new_grid = importlib.import_module(
        "exogibbs.equilibrium.gas.grid.service"
    )
    old_types = importlib.import_module(
        "exogibbs.optimize.fixed_support_v2.types"
    )
    new_types = importlib.import_module(
        "exogibbs.equilibrium.condensate.fixed_support.types"
    )
    old_batch = importlib.import_module(
        "exogibbs.optimize.fixed_support_v2_profile"
    )
    new_batch = importlib.import_module(
        "exogibbs.equilibrium.condensate.fixed_support.batch"
    )

    assert old_equations.compute_ln_normalized_pressure is (
        new_equations.compute_ln_normalized_pressure
    )
    assert old_solver.minimize_gibbs is new_solver.minimize_gibbs
    assert old_potential.gibbs_energies is new_potential.gibbs_energies
    assert old_grid is new_grid
    assert old_types is new_types
    assert old_batch.PreparedFixedSupportV2Bucket is (
        new_batch.PreparedFixedSupportV2Bucket
    )
    assert old_batch.run_fixed_support_profile is (
        new_batch.run_fixed_support_profile
    )

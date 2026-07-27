"""Production-route import and fallback contract tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def test_default_head_v2_does_not_import_or_fallback_to_legacy_solver() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        import jax.numpy as jnp

        from exogibbs.api.chemistry import ChemicalSetup

        module = importlib.import_module(
            "exogibbs.api.condensate_equilibrium"
        )
        gas_setup = ChemicalSetup(
            formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
            hvector_func=lambda temperature: jnp.asarray([0.0, 0.0]),
            elements=("H", "O"),
            species=("H", "O"),
        )
        condensate_setup = ChemicalSetup(
            formula_matrix=jnp.asarray([[2.0], [1.0]]),
            hvector_func=lambda temperature: jnp.asarray([0.0]),
            elements=("H", "O"),
            species=("H2O_s",),
        )
        setup = module.build_condensate_chemical_setup(
            gas_setup=gas_setup,
            condensate_setup=condensate_setup,
        )
        options = module.CondensateEquilibriumOptions()
        legacy_modules = (
            "exogibbs.optimize.minimize_cond",
            "exogibbs.optimize.pipm_rgie_cond",
        )
        imported_before_call = {
            name: name in sys.modules for name in legacy_modules
        }

        def fail_v2(*args, **kwargs):
            raise RuntimeError("sentinel-v2-failure")

        module._run_head_v2_profile = fail_v2
        try:
            module.condensate_equilibrium(
                setup,
                300.0,
                1.0,
                jnp.asarray([1.0, 1.0]),
                options=options,
            )
        except RuntimeError as error:
            propagated_error = str(error)
        else:
            propagated_error = None

        print(
            json.dumps(
                {
                    "exogibbs_file": importlib.import_module(
                        "exogibbs"
                    ).__file__,
                    "route": options.route,
                    "route_version": (
                        module.CONDENSATE_HEAD_V2_ROUTE_VERSION
                    ),
                    "preset": options.fixed_support_v2_preset,
                    "imported_before_call": imported_before_call,
                    "imported_after_call": {
                        name: name in sys.modules
                        for name in legacy_modules
                    },
                    "propagated_error": propagated_error,
                }
            )
        )
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
    )
    payload = json.loads(completed.stdout.splitlines()[-1])

    assert Path(payload["exogibbs_file"]).resolve() == (
        SOURCE_ROOT / "exogibbs" / "__init__.py"
    ).resolve()
    assert payload["route"] == "head_v2"
    assert payload["route_version"] == "v2.0"
    assert payload["preset"] == "validated_2026_07"
    assert payload["imported_before_call"] == {
        "exogibbs.optimize.minimize_cond": False,
        "exogibbs.optimize.pipm_rgie_cond": False,
    }
    assert payload["imported_after_call"] == payload["imported_before_call"]
    assert payload["propagated_error"] == "sentinel-v2-failure"


def test_prepared_v2_plan_uses_v2_owned_buckets_without_legacy_imports() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        import jax.numpy as jnp

        from exogibbs.api.chemistry import ChemicalSetup

        module = importlib.import_module(
            "exogibbs.api.condensate_equilibrium"
        )
        gas_setup = ChemicalSetup(
            formula_matrix=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
            hvector_func=lambda temperature: jnp.asarray([0.0, 0.0]),
            elements=("H", "O"),
            species=("H", "O"),
        )
        condensate_setup = ChemicalSetup(
            formula_matrix=jnp.asarray([[2.0], [1.0]]),
            hvector_func=lambda temperature: jnp.asarray([0.0]),
            elements=("H", "O"),
            species=("H2O_s",),
        )
        setup = module.build_condensate_chemical_setup(
            gas_setup=gas_setup,
            condensate_setup=condensate_setup,
        )
        init = tuple(
            module.CondensateEquilibriumInit(
                gas_ln_n=jnp.log(jnp.asarray([0.5, 0.5])),
                gas_ntot=jnp.asarray(1.0),
            )
            for _ in range(2)
        )
        legacy_modules = (
            "exogibbs.optimize.minimize_cond",
            "exogibbs.optimize.pipm_rgie_cond",
        )
        plan = module.prepare_experimental_profile_fixed_support_batch_plan(
            setup,
            jnp.asarray([300.0, 310.0]),
            jnp.asarray([1.0, 1.0]),
            jnp.asarray([1.0, 1.0]),
            support_indices=(0,),
            support_amounts_init=(0.1,),
            init=init,
        )
        print(
            json.dumps(
                {
                    "bucket_module": type(plan.buckets[0]).__module__,
                    "legacy_imports": {
                        name: name in sys.modules for name in legacy_modules
                    },
                }
            )
        )
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
    )
    payload = json.loads(completed.stdout.splitlines()[-1])

    assert payload["bucket_module"] == (
        "exogibbs.equilibrium.condensate.fixed_support.batch"
    )
    assert payload["legacy_imports"] == {
        "exogibbs.optimize.minimize_cond": False,
        "exogibbs.optimize.pipm_rgie_cond": False,
    }

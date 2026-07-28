import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples" / "comparisons"


def _guarded_main_call_count(tree: ast.Module) -> int:
    return sum(
        1
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "main"
            for statement in node.body
        )
    )


@pytest.mark.parametrize(
    ("filename", "chemistry_mode", "public_api"),
    [
        (
            "comparison_with_fastchem4_gas.py",
            '"gas"',
            "solve_gas_profile",
        ),
        (
            "comparison_with_fastchem4_condensates.py",
            '"equilibrium_condensation"',
            "solve_condensate_profile",
        ),
    ],
)
def test_fastchem4_comparison_example_is_current_and_main_guarded(
    filename,
    chemistry_mode,
    public_api,
):
    path = EXAMPLE_ROOT / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    compile(source, str(path), "exec")
    assert "pyfastchem" not in source
    assert "exojax" not in source
    assert "run_fastchem_executable" in source
    assert f"chemistry_mode={chemistry_mode}" in source
    assert public_api in source
    assert "build_aligned_abundance_vector" in source
    assert "occurrence_keys" in source
    assert "elements_conserved" in source

    assert _guarded_main_call_count(tree) == 1


@pytest.mark.parametrize(
    ("legacy_filename", "current_module"),
    [
        ("comparison_with_fastchem.py", "comparison_with_fastchem4_gas"),
        (
            "comparison_with_fastchem_extended.py",
            "comparison_with_fastchem4_gas",
        ),
        (
            "comparison_with_fastchem_cond.py",
            "comparison_with_fastchem4_condensates",
        ),
    ],
)
def test_historical_fastchem_example_path_delegates_to_current_example(
    legacy_filename,
    current_module,
):
    path = EXAMPLE_ROOT / legacy_filename
    source = path.read_text(encoding="utf-8")

    compile(source, str(path), "exec")
    assert f"from {current_module} import main" in source
    assert 'if __name__ == "__main__":' in source


def test_restored_fastchem_initializer_example_keeps_solver_inputs_independent():
    path = EXAMPLE_ROOT / "comparison_with_fastchem_initializer.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    compile(source, str(path), "exec")
    assert "pyfastchem" not in source
    assert "exojax" not in source
    assert "GridEquilibriumInitializer" in source
    assert "solve_gas_profile" in source
    assert "build_aligned_abundance_vector" in source
    assert "run_fastchem_executable" in source
    assert 'chemistry_mode="gas"' in source
    assert 'path="fastchem/logK/logK.dat"' in source
    assert source.index("grid_result, grid_diagnostics") < source.index(
        "fastchem = run_fastchem_executable"
    )
    assert source.index("uniform_result, uniform_diagnostics") < source.index(
        "fastchem = run_fastchem_executable"
    )
    assert _guarded_main_call_count(tree) == 1


@pytest.mark.parametrize(
    ("filename", "required_fragments"),
    [
        (
            "comparison_with_hsystem.py",
            (
                "from exogibbs.api.gas import EquilibriumOptions, solve",
                "HSystem",
                "jax.jacrev",
                "temperature_gradient_reference",
                "pressure_gradient_reference",
            ),
        ),
        (
            "comparison_with_hcosystem.py",
            (
                "from exogibbs.api.gas import EquilibriumOptions, solve",
                "HCOSystem",
                "jax.jacrev",
                "_bisect_analytic_co",
                "derivative_dlnnCO_db",
            ),
        ),
        (
            "comparison_with_ykcode.py",
            (
                "from exogibbs.api.gas import EquilibriumOptions, solve",
                "LEGACY_ELEMENT_BUDGET",
                "EXPECTED_SPECIES_ORDER_SHA256",
                "result.n",
                "MAX_RELATIVE_ERROR_LIMIT = 0.051",
            ),
        ),
    ],
)
def test_restored_trace_example_uses_current_api_and_main_guard(
    filename,
    required_fragments,
):
    path = EXAMPLE_ROOT / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    compile(source, str(path), "exec")
    assert "exogibbs.optimize" not in source
    for fragment in required_fragments:
        assert fragment in source
    assert _guarded_main_call_count(tree) == 1


def test_historical_ykb4_reference_payload_is_exact_and_well_formed():
    path = REPOSITORY_ROOT / "examples" / "data" / "p10.txt"
    payload = path.read_bytes()
    values = np.loadtxt(path, delimiter=",")

    assert hashlib.sha256(payload).hexdigest() == (
        "062a0d21768f85871b7980ae3883d34f2466b9e11618255060e19d32c4a8612b"
    )
    assert values.shape == (160,)
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)
    assert np.count_nonzero(values > 1.0e-14) == 16

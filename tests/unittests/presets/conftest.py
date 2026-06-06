"""Collection guards for optional FastChem4 milestone artifact tests."""

from __future__ import annotations

from pathlib import Path

import pytest


OPTIONAL_ARTIFACT_TEST_PREFIXES = (
    "fastchem4_milestone083_docs_package_contract_test.py",
    "fastchem4_milestone3821_to_3900_outer_recompute_tradeoff_and_itercap_test.py",
    "fastchem4_milestone3901_to_3960_persisted_support_condensed_stage_test.py",
    "fastchem4_milestone3976_to_3985_kkt_component_floor_audit_test.py",
    "fastchem4_milestone3986_to_3995_stationarity_frame_audit_test.py",
    "fastchem4_milestone4085_thermo_valid_support_helper_replay_test.py",
    "fastchem4_milestone4086_algorithm_v11_thermo_valid_callsite_trial_test.py",
    "fastchem4_milestone4087_algorithm_v11_multistep_convergence_probe_test.py",
    "fastchem4_milestone4088_gas_stationarity_metric_frame_decomposition_test.py",
    "fastchem4_milestone4089_amount_weighted_gas_acceptance_replay_test.py",
    "fastchem4_milestone4090_amount_weighted_remaining_residual_decomposition_test.py",
    "fastchem4_milestone4091_amount_weighted_state_cap_sweep_test.py",
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip optional milestone artifact checks when result artifacts are absent."""

    reason = (
        "Optional FastChem4 milestone artifact test requires results artifacts "
        "that are intentionally not committed."
    )
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        file_name = Path(str(item.fspath)).name
        if file_name in OPTIONAL_ARTIFACT_TEST_PREFIXES:
            item.add_marker(marker)

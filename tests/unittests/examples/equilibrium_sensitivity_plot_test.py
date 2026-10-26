"""Check the semantics of the optional response-network visualization."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


nx = pytest.importorskip("networkx")
pytest.importorskip("matplotlib")
EXAMPLE = Path(__file__).resolve().parents[3] / "examples/_equilibrium_sensitivity_plot.py"
SPEC = importlib.util.spec_from_file_location("equilibrium_sensitivity_plot", EXAMPLE)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def test_response_graph_preserves_sign_direction_and_isolated_nodes():
    response = np.zeros((5, 5))
    response[4, 1] = 1.25
    response[2, 2] = -0.25
    response[3, 4] = 0.149
    response[0, 0] = 0.15
    amounts = np.ones(5)

    pruned = PLOT.response_graph(response, amounts)
    complete = PLOT.response_graph(response, amounts, threshold=0.0)

    assert pruned.is_directed()
    assert nx.is_bipartite(pruned)
    assert set(pruned) == set(complete) == set(PLOT.CONTROLS + PLOT.SPECIES)
    assert set(pruned.edges) == {("C", "CH4"), ("O", "CO"), ("H", "H2")}
    assert pruned["O"]["CO"]["response"] == -0.25
    assert complete["K2"]["CO2"]["response"] == 0.149
    assert pruned.degree("K2") == pruned.degree("CO2") == 0
    assert all(pruned.nodes[source]["kind"] == "control" and
               pruned.nodes[target]["kind"] == "species" for source, target in pruned.edges)

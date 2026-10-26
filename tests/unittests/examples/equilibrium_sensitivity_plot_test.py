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


def _snapshot(temperature):
    reaction = np.array([[0.1, -0.02], [0.2, -0.05], [-0.3, -0.1], [-0.05, 0.8],
                         [0.01, -0.005] if temperature == 800 else [0.6, -0.02]])
    drive = np.array([[-30., 2.], [-5., 0.]]) if temperature == 800 else np.array([[-27., 2.], [-4., 0.]])
    return {"temperature": temperature, "reaction_response": reaction, "drive_response": drive,
            "physical_response": reaction @ drive,
            "contributions": reaction[:, :, None] * drive[None, :, :],
            "reductions": {"T": {"indices": [0, 1] if temperature == 800 else [0]},
                           "P": {"indices": [0]}}}


def _path_sum(graph, control):
    return sum(np.prod([graph[source][target]["response"] for source, target in zip(path, path[1:])])
               for path in nx.all_simple_paths(graph, control, "CH4"))


@pytest.mark.parametrize("as_lists", [False, True])
def test_physical_graph_is_direct_and_factorization_collapses_exactly(as_lists):
    snapshot = _snapshot(1000)
    if as_lists:
        snapshot = {key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in snapshot.items()}
    direct = PLOT.physical_control_graph(snapshot)
    factorized = PLOT.methane_control_graph(snapshot)
    collapsed = PLOT.methane_control_graph(snapshot, "collapsed")
    selected = PLOT.methane_control_graph(snapshot, "selected")

    assert nx.is_bipartite(direct)
    assert set(direct.edges) == {(control, species) for control in ("T", "P") for species in PLOT.SPECIES}
    assert set(collapsed.edges) == {("T", "CH4"), ("P", "CH4")}
    assert set(collapsed) == {"T", "P", "CH4"}
    assert not factorized.has_edge("P", "q2")  # Ideal-gas water-gas shift has delta_nu = 0.
    for control in ("T", "P"):
        assert not factorized.has_edge(control, "CH4")
        assert _path_sum(factorized, control) == pytest.approx(direct[control]["CH4"]["response"])
        assert collapsed[control]["CH4"]["response"] == pytest.approx(_path_sum(factorized, control))
    # The exact collapse includes the weak positive path discarded by selection.
    assert collapsed["T"]["CH4"]["response"] - _path_sum(selected, "T") == pytest.approx(0.08)


@pytest.mark.parametrize("temperature, expected", [(800, {"q1", "q2"}), (1000, {"q1"})])
def test_selected_graph_preserves_temperature_dependent_path_selection(temperature, expected):
    graph = PLOT.methane_control_graph(_snapshot(temperature), "selected")
    assert set(graph.successors("T")) == expected
    assert set(graph.successors("P")) == {"q1"}
    assert not graph.has_edge("T", "CH4")


def test_selected_graph_does_not_leak_one_controls_paths_into_another():
    snapshot = _snapshot(1000)
    snapshot["drive_response"][1, 1] = 1.0
    snapshot["reductions"]["P"]["indices"] = [1]
    graph = PLOT.methane_control_graph(snapshot, "selected")
    assert set(graph.edges) == {("T", "q1"), ("P", "q2"), ("q1", "CH4"), ("q2", "CH4")}

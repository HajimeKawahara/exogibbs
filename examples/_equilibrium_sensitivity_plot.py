"""Gas equilibrium response figures
================================

Figures for the gas-only equilibrium response example.

Plotting dependencies stay outside the numerical example's import path.
NetworkX holds the response graph; the Graphviz ``dot`` executable lays it out.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np


SPECIES = ("H2", "H2O", "CO", "CO2", "CH4")
CONTROLS = ("H", "C", "O", "K1", "K2")
SPECIES_LABELS = (r"H$_2$", r"H$_2$O", "CO", r"CO$_2$", r"CH$_4$")
CONTROL_LABELS = (r"$b_H$", r"$b_C$", r"$b_O$", r"$K_1$", r"$K_2$")
POSITIVE = "#2166ac"
NEGATIVE = "#c65d16"
EDGE_THRESHOLD = 0.15


def response_graph(
    response: np.ndarray,
    amounts: np.ndarray,
    threshold: float = EDGE_THRESHOLD,
) -> nx.DiGraph:
    """Keep signed total responses above a display threshold, retaining nodes."""
    graph = nx.DiGraph()
    fractions = amounts / np.sum(amounts)
    for name in CONTROLS:
        graph.add_node(name, kind="control", label=f"b_{name}" if len(name) == 1 else name)
    for name, fraction in zip(SPECIES, fractions):
        graph.add_node(name, kind="species", label=f"{name}\nx = {fraction:.2e}")
    for row, species in enumerate(SPECIES):
        for col, control in enumerate(CONTROLS):
            coefficient = float(response[row, col])
            if abs(coefficient) >= threshold:
                graph.add_edge(control, species, response=coefficient)
    return graph


def _save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _temperature_figure(
    output: Path,
    temperatures: np.ndarray,
    amounts: np.ndarray,
    responses: np.ndarray,
    thermal_contributions: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.4), constrained_layout=True)
    fractions = amounts / amounts.sum(axis=1, keepdims=True)
    for index, label in enumerate(SPECIES_LABELS):
        axes[0, 0].semilogy(temperatures, fractions[:, index], label=label, lw=2)
    axes[0, 0].set(title="Composition: methane gives way to CO", ylabel="Mole fraction x")
    axes[0, 0].legend(ncol=3, fontsize=9)
    for index, label in enumerate(CONTROL_LABELS[:3]):
        axes[0, 1].plot(temperatures, responses[:, 4, index], label=label, lw=2)
    axes[0, 1].set(
        title="Methane response to elemental budgets",
        ylabel=r"$\partial\ln n_{\rm CH_4}/\partial\ln b_e$",
    )
    axes[0, 1].legend(ncol=3)
    reaction_colors = (POSITIVE, NEGATIVE)
    for index, (label, color) in enumerate(zip(("K1: methanation", "K2: water-gas shift"), reaction_colors)):
        axes[1, 0].plot(temperatures, responses[:, 4, index + 3], label=label, color=color, lw=2)
        axes[1, 1].plot(temperatures, thermal_contributions[:, 4, index], label=label, color=color, lw=2)
    axes[1, 0].set(
        title="Which equilibrium condition matters for methane?",
        ylabel=r"$\partial\ln n_{\rm CH_4}/\partial\ln K_r$",
    )
    axes[1, 1].plot(
        temperatures, thermal_contributions[:, 4].sum(axis=1),
        "k--", label="Sum: total temperature response", lw=1.5,
    )
    axes[1, 1].set(
        title="Temperature response, resolved by equilibrium",
        ylabel=r"$(\partial\ln n_{\rm CH_4}/\partial\ln K_r)\,(d\ln K_r/d\ln T)$",
    )
    for axis in axes.flat:
        axis.set_xlabel("Temperature [K]")
        axis.grid(alpha=0.2)
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.axhline(0, color="0.5", lw=0.7)
    for axis in axes[1]:
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Gas-only H-C-O equilibrium | P = 1 bar | b(H,C,O) = (1, 0.03, 0.06)", fontsize=13)
    _save_figure(figure, output, "temperature_response")


def _matrix_figure(output: Path, snapshot: dict) -> None:
    response = np.asarray(snapshot["response"])
    maximum = float(np.max(np.abs(response)))
    colormap = LinearSegmentedColormap.from_list("response", [NEGATIVE, "white", POSITIVE])
    figure, axis = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    plotted = axis.imshow(response, cmap=colormap, vmin=-maximum, vmax=maximum)
    for row in range(5):
        for col in range(5):
            value = response[row, col]
            label = f"{value:+.3f}" if abs(value) >= 0.0005 else f"{value:+.1e}"
            axis.text(col, row, label, ha="center", va="center", fontsize=10,
                      color="white" if abs(value) > maximum * 0.65 else "black")
    axis.set(
        xticks=np.arange(5), xticklabels=CONTROL_LABELS,
        yticks=np.arange(5), yticklabels=SPECIES_LABELS,
        xlabel="Perturbed control (all others held fixed)", ylabel="Responding species amount",
        title=f"All local responses at {snapshot['temperature']:g} K\nNo display threshold applied",
    )
    axis.axvline(2.5, color="black", lw=1.2)
    figure.colorbar(plotted, ax=axis, label=r"$\partial\ln n_i / \partial\ln u_j$")
    _save_figure(figure, output, "response_matrix")


def _network_figure(output: Path, snapshots: list[dict]) -> None:
    quote = json.dumps
    lines = [
        "digraph responses {",
        'graph [rankdir=LR, bgcolor="white",',
        'fontname="DejaVu Sans", fontsize=16, labelloc=t, nodesep=0.34, ranksep=1.0,',
        'label="Local equilibrium responses: controls -> species amounts\\n'
        'Blue = positive; orange dashed = negative; labels = d ln(n) / d ln(control)\\n'
        '|response| >= 0.15 shown; width = 0.7 + 1.3 min(|response|, 3). '
        'Display pruning only; arrows are not reaction paths."];',
        'node [fontname="DejaVu Sans", fontsize=11, margin="0.10,0.07", style=filled];',
        'edge [fontname="DejaVu Sans", fontsize=10, arrowsize=0.65];',
    ]
    for index, snapshot in enumerate(snapshots):
        graph = response_graph(np.asarray(snapshot["response"]), np.asarray(snapshot["amounts"]))
        lines.extend([
            f"subgraph cluster_{index} {{",
            f'label="{snapshot["temperature"]:g} K | {graph.number_of_edges()}/25 responses shown";',
            'color="#d0d5db"; style=rounded; margin=18;',
        ])
        node_id = lambda name: quote(f"s{index}_{name}")
        for name, attributes in graph.nodes(data=True):
            is_control = attributes["kind"] == "control"
            shape, fill = ("box", "#eef2f6") if is_control else ("ellipse", "#fff9e8")
            highlight = ', color="#6a3d9a", penwidth=2.5, fontname="DejaVu Sans Bold"' if name == "CH4" else ""
            fill = "#eee7f5" if name == "CH4" else fill
            lines.append(f'{node_id(name)} [label={quote(attributes["label"])}, shape={shape}, fillcolor="{fill}"{highlight}];')
        for names in (CONTROLS, SPECIES):
            lines.append("{ rank=same; " + "; ".join(node_id(name) for name in names) + "; }")
            lines.append(" -> ".join(node_id(name) for name in reversed(names)) + " [style=invis, weight=100];")
        # Anchor both columns independently of which response edges survive.
        for control, species in zip(CONTROLS, SPECIES):
            lines.append(f'{node_id(control)} -> {node_id(species)} [style=invis, weight=1000, minlen=3];')
        for source, target, attributes in graph.edges(data=True):
            value = attributes["response"]
            color = POSITIVE if value > 0 else NEGATIVE
            style = "solid" if value > 0 else "dashed"
            width = 0.7 + 1.3 * min(abs(value), 3.0)
            lines.append(
                f'{node_id(source)} -> {node_id(target)} [label="{value:+.2f}", '
                f'color="{color}", fontcolor="{color}", style={style}, constraint=false, penwidth={width:.2f}];'
            )
        lines.append("}")
        if index:
            # Invisible links arrange panels without Graphviz's cluster packer.
            lines.append(f'"s{index - 1}_H2" -> "s{index}_H" [style=invis, minlen=2, weight=100];')
    lines.append("}")
    _write_dot(output, "sensitivity_network", lines)


def _write_dot(output: Path, stem: str, lines: list[str]) -> None:
    dot_path = output / f"{stem}.dot"
    dot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for suffix in ("png", "pdf", "svg"):
        subprocess.run(
            ["dot", f"-T{suffix}", "-Gdpi=150", str(dot_path), "-o", str(dot_path.with_suffix(f".{suffix}"))],
            check=True,
        )


def physical_control_graph(snapshot: dict) -> nx.DiGraph:
    """Return all direct log-temperature/pressure responses, with no pruning."""
    graph = nx.DiGraph()
    graph.add_nodes_from((name, {"label": name}) for name in ("T", "P") + SPECIES)
    response = np.asarray(snapshot["physical_response"])
    for i, species in enumerate(SPECIES):
        for j, control in enumerate(("T", "P")):
            graph.add_edge(control, species, response=float(response[i, j]))
    return graph


def methane_control_graph(
    snapshot: dict, mode: Literal["full", "selected", "collapsed"] = "full",
) -> nx.DiGraph:
    """Represent one view of the methane chain rule without double counting."""
    if mode not in ("full", "selected", "collapsed"):
        raise ValueError("mode must be full, selected, or collapsed")
    graph = nx.DiGraph()
    graph.add_nodes_from((name, {"label": name}) for name in ("T", "P", "CH4"))
    if mode == "collapsed":
        total = np.asarray(snapshot["contributions"])[4].sum(axis=0)
        for control, value in zip(("T", "P"), total):
            graph.add_edge(control, "CH4", response=float(value))
        return graph
    selected = snapshot["reductions"]
    reactions = (sorted(set(r for control in ("T", "P") for r in selected[control]["indices"]))
                 if mode == "selected" else [0, 1])
    response, drive = np.asarray(snapshot["reaction_response"]), np.asarray(snapshot["drive_response"])
    for r in reactions:
        name = f"q{r + 1}"
        graph.add_node(name, label=name + "\n" + ("methanation" if r == 0 else "water-gas shift"))
        graph.add_edge(name, "CH4", response=float(response[4, r]))
        for j, control in enumerate(("T", "P")):
            if drive[r, j] != 0 and (mode == "full" or r in selected[control]["indices"]):
                graph.add_edge(control, name, response=float(drive[r, j]))
    return graph


def _control_panels(output: Path, stem: str, title: str, panels: list[tuple]) -> None:
    quote = json.dumps
    lines = ["digraph response {", "graph [rankdir=LR, bgcolor=white, nodesep=0.5, ranksep=0.8,",
             f'fontname="DejaVu Sans", fontsize=16, labelloc=t, label={quote(title)}];',
             'node [fontname="DejaVu Sans", fontsize=13, style=filled, margin="0.15,0.09"];',
             'edge [fontname="DejaVu Sans", fontsize=12, arrowsize=0.75];']
    previous = None
    for number, (caption, graph, ranks) in enumerate(panels):
        ident = lambda name: quote(f"p{number}_{name}")
        lines += [f"subgraph cluster_{number} {{", f"label={quote(caption)};",
                  'color="#d0d5db"; style=rounded; margin=20;']
        for name, attrs in graph.nodes(data=True):
            shape, fill = ("box", "#eef2f6") if name in ("T", "P") else ("ellipse", "#fff9e8")
            extra = ', color="#6a3d9a", penwidth=2.3, fontname="DejaVu Sans Bold"' if name == "CH4" else ""
            fill = "#eee7f5" if name == "CH4" else fill
            lines.append(f'{ident(name)} [label={quote(attrs["label"])}, shape={shape}, fillcolor="{fill}"{extra}];')
        for rank in ranks:
            lines.append("{rank=same; " + "; ".join(ident(name) for name in rank) + ";}")
        for source, target, attrs in graph.edges(data=True):
            value = attrs["response"]
            color, style = (POSITIVE, "solid") if value >= 0 else (NEGATIVE, "dashed")
            width = 0.8 + 0.12 * min(abs(value), 25)
            lines.append(f'{ident(source)} -> {ident(target)} [label="{value:+.5g}", '
                         f'color="{color}", fontcolor="{color}", style={style}, penwidth={width:.3f}];')
        lines.append("}")
        if previous:
            lines.append(f'{previous} -> {ident(ranks[0][0])} [style=invis, minlen=2, weight=100];')
        previous = ident(ranks[-1][0])
    lines.append("}")
    _write_dot(output, stem, lines)


def render_physical_controls(output: Path, snapshots: list[dict], retention: float = 0.99) -> None:
    """Write physical controls and distinct exact/approximate methane reductions."""
    output.mkdir(parents=True, exist_ok=True)
    snapshot = next(row for row in snapshots if row["temperature"] == 1000)
    controls = ["T", "P"]
    _control_panels(output, "physical_controls",
                    "Direct physical controls at 1000 K, P = 1 bar; elemental budgets fixed\n"
                    "Edges: d ln(n_i) / d ln(T or P); blue positive, orange dashed negative\n"
                    "Width = 0.8 + 0.12 min(|response|, 25); all ten responses shown",
                    [("Bipartite response graph", physical_control_graph(snapshot), [controls, SPECIES])])
    panels = []
    for mode in ("full", "selected"):
        graph = methane_control_graph(snapshot, mode)
        if mode == "selected":
            reduction = snapshot["reductions"]["T"]
            caption = (f"B. Approximate: keep >={100 * retention:g}% of absolute path contributions\n"
                       f"T retained: {reduction['retained']:+.5f}; omitted: {reduction['omitted']:+.5f}\n"
                       f"Retained absolute fraction: {100 * reduction['retained_absolute_fraction']:.3f}%")
        else:
            products = np.asarray(snapshot["contributions"])[4, :, 0]
            caption = f"A. Full two-equilibrium basis\nT path products: {products[0]:+.5f}, {products[1]:+.5f}"
        panels.append((caption, graph, [controls, [name for name in graph if name.startswith("q")], ["CH4"]]))
    panels.append(("C. Exact collapse\nSigned sum over ALL path products",
                   methane_control_graph(snapshot, "collapsed"), [controls, ["CH4"]]))
    _control_panels(output, "methane_reduction",
                    "Methane response at 1000 K: factorization, selection, exact collapse\n"
                    "q = ln K(T) - delta_nu ln(P/P0); pressure changes q, not ideal-gas K\n"
                    "A/B edges: dq / d ln(T or P), then d ln(n_CH4) / dq; C: d ln(n_CH4) / d ln(T or P)\n"
                    "Blue positive; orange dashed negative. Separate views; do not add direct and factorized edges.",
                    panels)


def render_figures(
    output: Path,
    temperatures: np.ndarray,
    amounts: np.ndarray,
    responses: np.ndarray,
    thermal_contributions: np.ndarray,
    snapshots: list[dict],
) -> None:
    """Write overview, complete response matrix, and signed sparse networks."""
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        _temperature_figure(output, temperatures, amounts, responses, thermal_contributions)
        _matrix_figure(output, snapshots[len(snapshots) // 2])
    _network_figure(output, snapshots)

"""Verify archived file identities and independent finite element balances."""

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(name):
    return json.loads((HERE / name).read_text())


def balance(matrix, amounts, budget):
    formula = np.asarray(matrix)
    n = np.asarray(amounts)
    b = np.asarray(budget)
    assert np.all(np.isfinite(n)) and np.all(n >= 0)
    total = formula @ n
    assert np.all(total[b == 0] == 0)
    error = float(np.max(np.abs(total[b > 0] / b[b > 0] - 1)))
    assert error < 1e-9
    return error


def main():
    receipt = read("receipt.json")
    for name, expected in receipt["file_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    errors = {}
    h = read("hydrogen.json")
    assert h["accepted"]
    errors["hydrogen"] = balance(h["ledger"]["formula_matrix"], h["state"]["component_amounts_mol"], h["state"]["element_amounts_mol"])
    for name in ("melts_present", "melts_absent"):
        r = read(name + ".json")
        assert r["result"]["accepted"]
        errors[name] = balance(r["formula_matrix"], r["result"]["component_amounts_mol"], r["element_amounts_mol"])
        assert np.max(np.abs(r["result"]["reaction_residual"])) < 1e-8
        if name.endswith("absent"):
            assert all(n == 0 for s, n in zip(r["component_order"], r["result"]["component_amounts_mol"]) if s.endswith("_metal"))
    reference = json.loads((ROOT / "examples/metal_silicate/sulfur_reference.json").read_text())
    networks = {n["model_id"]: n for n in reference["networks"].values()}
    for case in read("sulfur_carbon_source.json")["cases"]:
        assert case["accepted"]
        network = networks[case["model_id"]]
        formula = [[network["component_formulas"][s].get(e, 0) for s in case["species"]] for e in case["elements"]]
        errors[f'{case["model_id"]}_{case["T_K"]}'] = balance(formula, case["component_amounts_mol"], case["element_amounts_mol"])
        assert np.max(np.abs(case["reaction_residual"])) < 1e-8
    s = read("sulfide.json")
    assert any(state["accepted"] for state in s["states"])
    for i, state in enumerate(s["states"]):
        errors[f"sulfide_{i}"] = balance(s["formula_matrix_element_rows"], state["amounts_mol"], s["element_amounts_mol"])
        if state["accepted"]:
            assert state["max_reaction_residual"] < 1e-8
            assert abs(state["saturation_log_ratio"]) < 1e-8
    print(json.dumps({"max_relative_element_residual": max(errors.values()), "cases": errors}, indent=2))


if __name__ == "__main__":
    main()

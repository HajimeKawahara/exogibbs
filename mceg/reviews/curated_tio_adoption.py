"""Replay versioned TiO in both networks; its original default run is archived."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from exogibbs.api.gas import EquilibriumOptions, solve_profile
from exogibbs.presets import curated_c2h, curated_tio


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "tests/data/validation/tio_equilibrium_impact/20261103/equilibria.npz"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run(output: Path) -> None:
    """Cold-solve 75 cases per network and retain every species and constraint."""
    jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    with np.load(PREVIOUS) as previous:
        for ions in (True, False):
            setup, legacy = curated_tio.chemsetup(ions=ions), curated_c2h.chemsetup(ions=ions)
            assert setup.metadata["dataset"] == "curated-tio-v1"
            np.testing.assert_array_equal(setup.formula_matrix, legacy.formula_matrix)
            np.testing.assert_array_equal(setup.element_vector_reference, legacy.element_vector_reference)
            assert setup.species == legacy.species and setup.elements == legacy.elements
            t, p = previous["temperature_K"], previous["pressure_bar"]
            rows = [list(previous["elements"]).index(name) for name in setup.elements]
            budgets = previous["budget"][:, rows]
            h = np.asarray(setup.hvector_func(t))
            tio = setup.species.index("O1Ti1")
            delta = h - np.asarray(legacy.hvector_func(t))
            np.testing.assert_allclose(np.delete(delta, tio, axis=-1), 0, atol=1e-12, rtol=0)
            assert np.all(delta[:, tio] > 0)
            columns = [list(previous["species"]).index(name) for name in setup.species]
            np.testing.assert_allclose(h, previous["h"][1][:, columns], atol=1e-12, rtol=0)
            amounts, flags = [], []
            for start in range(0, len(t), 25):
                end = start + 25
                result, info = solve_profile(
                    setup, T=jnp.asarray(t[start:end]), P=jnp.asarray(p[start:end]),
                    b=jnp.asarray(budgets[start]),
                    options=EquilibriumOptions(epsilon_crit=5e-14, method="vmap_cold"),
                    return_diagnostics=True)
                amounts.append(np.asarray(result.ln_n))
                flags.append(np.asarray(info["converged"]))
            ln_n, converged = np.concatenate(amounts), np.concatenate(flags)
            assert np.all(converged) and np.isfinite(ln_n).all()
            n = np.exp(ln_n)
            ln_x = ln_n - logsumexp(ln_n, axis=-1, keepdims=True)
            a = np.asarray(setup.formula_matrix)
            positive = budgets[0] > 0
            element = np.max(abs(n @ a.T - budgets)[:, positive] / budgets[:, positive])
            references = [setup.species.index("e1-" if el == "e-" else el+"1") for el in setup.elements]
            logp = ln_x + np.log(p)[:, None]
            mass_action = np.max(abs(logp - logp[:, references] @ a + h))
            assert element < 1e-5 and mass_action < 1e-8
            record = {"ions": ions, "cases_passed": len(t), "species": len(setup.species),
                      "max_element_relative_residual": float(element),
                      "max_mass_action_abs_ln": float(mass_action)}
            if ions:
                charge = a[setup.elements.index("e-")]
                residual = np.max(abs(n @ charge) / (n @ abs(charge)))
                assert residual < 1e-12
                np.testing.assert_allclose(h, previous["h"][1], atol=1e-12, rtol=0)
                reference = previous["ln_n"][1].copy()
                reference -= logsumexp(reference, axis=-1, keepdims=True)
                error = np.max(abs(ln_x - reference))
                assert error < 1e-4
                record.update(max_charge_relative_residual=float(residual),
                              max_abs_delta_ln_x_from_candidate=float(error))
            name = "ions" if ions else "neutral"
            np.savez_compressed(output / f"{name}.npz", ln_n=ln_n, ln_x=ln_x, h=h,
                                formula_matrix=a, budget=budgets, species=setup.species,
                                elements=setup.elements, temperature_K=t, pressure_bar=p,
                                C_over_O=previous["C_over_O"], converged=converged)
            summaries.append(record)
            print(json.dumps(record), flush=True)
    summary = {"dataset": "curated-tio-v1", "legacy_dataset": "curated-c2h-v1",
               "epsilon_crit": 5e-14, "method": "vmap_cold", "networks": summaries,
               "scope": "Versioned TiO replay and numerical closure in both networks; ion mode also "
                        "replays the archived candidate. No new independent-solver or physical validation."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    paths = [Path(__file__), PREVIOUS]
    paths += [ROOT / "src/exogibbs/presets" / f"{name}.py"
              for name in ("curated_tio", "curated_c2h", "curated_v1", "fastchem4", "fastchem")]
    paths += [ROOT / "src/exogibbs/data/curated" / f"{name}.json" for name in ("gas_v1", "c2h_v1", "tio_v1")]
    paths += [ROOT / name for name in ("src/exogibbs/thermo/standard.py", "src/exogibbs/thermo/models.py",
              "src/exogibbs/data/FastChem4/logK/logK.dat",
              "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat")]
    paths += sorted((ROOT / "src/exogibbs/equilibrium/gas").rglob("*.py"))
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "result": "passed",
               "python": platform.python_version(), "numpy": np.__version__, "jax": jax.__version__,
               "backend": jax.default_backend(), "dtype": "float64", "command": " ".join(sys.argv),
               "selected_file_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths},
               "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir())}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)

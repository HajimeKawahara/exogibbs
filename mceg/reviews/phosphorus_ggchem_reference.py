"""Run unmodified GGChem16 with an independent NASA9-to-Kp adapter.

The adapter supplies sampled formation constants through GGChem's existing
constant fit type. This tests the solver with shared thermochemistry, not
GGChem's default database selection. No compiler is required by unit tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import numpy as np
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "results/phosphorus_cold/ggchem"
DRIVER = Path(__file__).resolve().with_name("phosphorus_ggchem_driver.f90")
SOURCES = ("datamod.f", "gauss16.f", "stindex.f", "is_nan.F", "smchem16.f")
FLAGS = ("-fdefault-real-8", "-fdefault-double-8", "-O2", "-std=legacy")
FIXTURES = ROOT / "tests/data/validation/phosphorus_cold"


def build_reference(work: Path = WORK) -> Path:
    """Build the five pinned upstream units and the separate adapter."""
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    identity = json.loads((FIXTURES / "ggchem_build.json").read_text())
    source_zip = FIXTURES / "ggchem_source.zip"
    if hashlib.sha256(source_zip.read_bytes()).hexdigest() != identity["source_zip_sha256"]:
        raise ValueError("Vendored GGChem source ZIP changed")
    allowed = {"LICENSE", "README.md", *["src16/" + n for n in SOURCES]}
    for record in identity["files"]:
        if record["path"] not in allowed:
            raise ValueError("Unexpected GGChem source path")
        path = work / "upstream" / record["path"]
        if not path.exists():
            with zipfile.ZipFile(source_zip) as archive:
                content = archive.read(record["path"])
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"Vendored GGChem source changed: {record['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"GGChem source changed: {path}")
    source_paths = [str(work / "upstream/src16" / name) for name in SOURCES]
    binary = work / "phosphorus_ggchem"
    commands = [["gfortran", *FLAGS, "-c", *source_paths],
                ["gfortran", *FLAGS, str(DRIVER), *[Path(n).stem + ".o" for n in SOURCES], "-o", str(binary)]]
    for command in commands:
        subprocess.run(command, cwd=work, check=True, capture_output=True, text=True)
    identity.update({"adapter_sha256": hashlib.sha256(DRIVER.read_bytes()).hexdigest(),
                     "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
                     "build_commands": [[v.replace(str(ROOT), "$EXOGIBBS_ROOT") for v in c] for c in commands],
                     "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                     "scope": "Unmodified SMCHEM16; independent NASA9 evaluation and pressure iteration in the adapter."})
    (work / "build.json").write_text(json.dumps(identity, indent=2) + "\n")
    return binary


def run_reference(data: dict, names: list, temperature_K: np.ndarray,
                  pressure_bar: float, budget: dict, *, binary: Path | None = None,
                  output_directory: Path | None = None) -> dict:
    """Return log mole fractions in the requested order and independent closure."""
    binary = WORK / "phosphorus_ggchem" if binary is None else Path(binary)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=WORK)) if output_directory is None else Path(output_directory)
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    elements = list(budget)
    records = data["records"]
    atoms = [name for name in elements if name in names and records[name]["composition"] == {name: 1}]
    if atoms != elements or budget["H"] <= 0:
        raise ValueError("The reference run needs every budget atom and positive hydrogen")
    molecules = [name for name in names if name not in atoms]
    order = atoms + molecules
    t = np.atleast_1d(np.asarray(temperature_K, dtype=float))
    p = np.broadcast_to(np.asarray(pressure_bar, dtype=float), t.shape)
    lines = [f"{len(atoms)} {len(molecules)} {len(t)}"]
    for name in order:
        record = records[name]
        values = ([budget[name]] if name in atoms else
                  [record["composition"].get(e, 0) for e in elements])
        lines.append(" ".join([name.upper() if name not in atoms else name,
                               *[str(v) for v in values], str(len(record["intervals"]))]))
        for interval in record["intervals"]:
            lines.append(" ".join(format(v, ".17g") for v in interval["bounds_K"] + interval["coefficients"]))
    lines.extend(f"{tk:.17g} {pk:.17g}" for tk, pk in zip(t, p))
    input_path, output_path = directory / "input.txt", directory / "output.txt"
    input_path.write_text("\n".join(lines) + "\n")
    run = subprocess.run([str(binary.resolve()), str(input_path.resolve()), str(output_path.resolve())],
                         cwd=directory, capture_output=True, text=True, timeout=120)
    (directory / "solver.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    rows = output_path.read_text().splitlines()
    if len(rows) != 1 + 4*len(t):
        raise RuntimeError(f"GGChem did not finish all requested cases: {directory}")
    header = np.array([[float(v) for v in rows[1+4*i].split()] for i in range(len(t))])
    ln_density = np.array([[float(v) for row in rows[2+4*i:4+4*i] for v in row.split()] for i in range(len(t))])
    ln_k = np.array([[0.]*len(atoms) + [float(v) for v in rows[4+4*i].split()] for i in range(len(t))])
    logx = ln_density - logsumexp(ln_density, axis=1)[:, None]
    matrix = np.array([[records[s]["composition"].get(e, 0) for s in order] for e in elements])
    logp = logx + np.log(p)[:, None]
    residual = logp - logp[:, :len(atoms)] @ matrix - ln_k
    eps = np.array(list(budget.values())) / budget["H"]
    closure = np.exp(ln_density - np.log(header[:, 2])[:, None]) @ matrix.T
    indices = [order.index(name) for name in names]
    return {"ln_mole_fraction": logx[:, indices], "ln_formation_Kp": ln_k[:, indices],
            "element_relative_residual": float(np.max(np.abs(closure/eps - 1))),
            "mass_action_max_abs_ln": float(np.max(np.abs(residual))),
            "pressure_relative_residual": float(np.max(np.abs(header[:, 3]))),
            "max_pressure_iterations": int(np.max(header[:, 4])),
            "output_directory": str(directory.relative_to(ROOT))}

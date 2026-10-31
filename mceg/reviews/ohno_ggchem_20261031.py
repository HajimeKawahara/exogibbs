"""Run and export a pinned native GGChem database comparison for Ohno's p3.

The two selections are GGChem's JANAF-based baseline and a final full-Burcat
override. They are mixed-source selections, not two pure database exports.
The optional instrumentation writes constants and primitive densities only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import numpy as np
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/ohno_p3"
REVISION = "65bbb8f9ca3d6c0b4e58cb7dd3862322452d0d4b"
ARCHIVE_SHA256 = "4eb41336b01919f101496473c15e1d31ff9242422e220f00910d6aa73e0044cd"
UPSTREAM = OUTPUT / f"GGchem-{REVISION}"
ABUNDANCES = ROOT / "src/exogibbs/data/FastChem4/element_abundances/asplund_2021.dat"
BASE_FILES = ["dispol_BarklemCollet.dat", "dispol_StockKitzmann_withoutTsuji.dat",
              "dispol_WoitkeRefit.dat"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_binary() -> Path:
    """Build original and output-instrumented executables from the pinned ZIP."""
    source = OUTPUT / "ggchem_upstream.zip"
    if digest(source) != ARCHIVE_SHA256:
        raise ValueError("Pinned GGChem archive hash mismatch")
    with zipfile.ZipFile(source) as archive:
        archive.extractall(OUTPUT)
    directory = UPSTREAM / "src16"
    command = ["make", "-f", "makefile.prodimo", "-j1"]
    with (OUTPUT / "native_build_original.log").open("w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(directory / "ggchem16", UPSTREAM / "ggchem16_original")
    path = directory / "smchem16.f"
    text = path.read_text()
    anchor = "      g(TiC) = -2.30256*arg\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + """
*     Diagnostic output only: final constants after the TiC exception.
      open(92,file='native_logk.dat',status='unknown',
     &     position='append')
      write(92,'(*(ES27.18E4,1X))') Tg,(g(i),i=1,nml)
      close(92)
""")
    path.write_text(text)
    path = directory / "demo_sweep.f"
    text = path.read_text()
    anchor = "        if (TEAinterface) then\n          nTEA = 0.0"
    assert text.count(anchor) == 1
    text = text.replace(anchor, """*       Diagnostic output only: retain the native quad-precision tail.
        open(91,file='native_logdensity.dat',status='unknown',
     &       position='append')
        write(91,'(*(ES27.18E4,1X))') Tg,nHges,pgas,
     &       (LOG(MAX(1.Q-4900,nat(elnum(jj)))),jj=1,el-1),
     &       (LOG(MAX(1.Q-4900,nat(elnum(jj)))),jj=el+1,NELM),
     &       (LOG(MAX(1.Q-4900,nmol(jj))),jj=1,NMOLE)
        close(91)

""" + anchor)
    path.write_text(text)
    with (OUTPUT / "native_build_instrumented.log").open("w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    return directory / "ggchem16"


def abundances() -> dict:
    return {parts[0]: float(parts[1]) for line in ABUNDANCES.read_text().splitlines()
            if (parts := line.split()) and not line.startswith("#") and parts[0] != "e-"}


def parse_record_header(line: str) -> tuple:
    """Read native fixed-width output, including an adjacent token like Cl12."""
    n = int(line[20:22])
    elements = [line[24+3*j:26+3*j].strip() for j in range(n)]
    start = 23 + 3*n
    amounts = [int(line[start+3*j:start+3*j+2]) for j in range(n)]
    return line[:20].strip(), dict(zip(elements, amounts))


def run_native(binary: Path, run: Path, elements: list, burcat: bool) -> None:
    """Use native input parsing, source overrides, sweep and quad solver."""
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "data").exists():
        (run / "data").symlink_to(UPSTREAM / "data", target_is_directory=True)
    (run / "abundances.dat").write_text("\n".join(f"{e} {v}" for e, v in abundances().items()) + "\n")
    files = BASE_FILES + (["dispol_BURCAT.dat"] if burcat else [])
    lines = [" ".join(elements)]
    lines += [f"{name} ! dispol_file{'' if i == 0 else i+1}" for i, name in enumerate(files)]
    lines += ["0 ! abund_pick", "abundances.dat", ".false. ! model_eqcond",
              ".false. ! remove_condensates", ".false. ! useDatabase",
              ".true. ! output_dispol", ".true. ! initchem_info", "1 ! model_dim",
              ".true. ! model_pconst", "3000. ! Tmax", "200. ! Tmin", ".001 ! pmax",
              ".001 ! pmin", "121 ! Npoints", "1.e10 ! Tfast", "-2 ! verbose"]
    (run / "input.in").write_text("\n".join(lines) + "\n")
    for name in ("native_logk.dat", "native_logdensity.dat"):
        (run / name).unlink(missing_ok=True)
    with (run / "run.log").open("w") as log:
        subprocess.run([str(binary), "input.in"], cwd=run, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    if len((run / "Static_Conc.dat").read_text().splitlines()) != 124:
        raise RuntimeError(f"Native GGChem did not finish 121 states: {run}")


def read_case(run: Path) -> dict:
    """Keep native molecular indices, including ambiguous original labels."""
    elements = (run / "input.in").read_text().splitlines()[0].split()
    eps = np.array([10.**(abundances()[e]-12.) for e in elements])
    lines = (run / "dispol.out").read_text().splitlines()
    headers = [parse_record_header(lines[1+2*i]) for i in range(int(lines[0]))]
    labels = [name for name, _ in headers]
    counts = Counter(labels)
    sources = {int(m.group(1)): m.group(3) for m in re.finditer(
        r"^\s*(\d+)\s+(\S+)\s+ ->\s+(\S+)\s*$", (run / "run.log").read_text(), re.M)}
    species, compositions, records = list(elements), [{e: 1} for e in elements], {}
    for i, label in enumerate(labels):
        composition = headers[i][1]
        name = label if counts[label] == 1 else f"{label}@{i+1}"
        species.append(name)
        compositions.append(composition)
        values = lines[2+2*i].split()
        records[name] = {"native_molecule_index": i+1, "display_name": label,
                         "composition": composition, "fit": int(values[0]),
                         "coefficients": list(map(float, values[1:])), "source": sources[i+1]}
    matrix = np.array([[c.get(e, 0) for c in compositions] for e in elements])
    density = np.loadtxt(run / "native_logdensity.dat")
    constants = np.loadtxt(run / "native_logk.dat")
    indices = [np.flatnonzero(np.isclose(constants[:, 0], t, rtol=1e-13))[-1] for t in density[:, 0]]
    ln_k = np.concatenate((np.zeros((len(indices), len(elements))), constants[indices, 1:]), axis=1)
    ln_k -= (1-matrix.sum(axis=0)) * np.log(1e6)
    logx = density[:, 3:] - logsumexp(density[:, 3:], axis=1)[:, None]
    logp = logx + np.log(.001)
    mass_action = logp - logp[:, :len(elements)] @ matrix - ln_k
    conservation = np.exp(density[:, 3:]-np.log(density[:, 1])[:, None]) @ matrix.T / eps - 1.
    closure = {"max_element_relative_residual": float(np.max(abs(conservation))),
               "max_mass_action_abs_ln": float(np.max(abs(mass_action))),
               "max_pressure_relative_residual": float(np.max(abs(density[:, 2]/1000.-1)))}
    if (closure["max_element_relative_residual"] > 1e-8 or
            closure["max_mass_action_abs_ln"] > 1e-7 or closure["max_pressure_relative_residual"] > 1e-8):
        raise RuntimeError(f"Native closure failed: {run}, {closure}")
    return {"elements": elements, "species": species, "formula_matrix": matrix.tolist(),
            "budget": eps.tolist(), "temperature_K": density[:, 0].tolist(), "pressure_bar": .001,
            "lnK_bar": ln_k.tolist(), "ggchem_log_mole_fractions": logx.tolist(),
            "effective_records": records, "name_collisions": {s: n for s, n in counts.items() if n > 1},
            "native_closure": closure,
            "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                             (run / "input.in", run / "abundances.dat", run / "dispol.out")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Rebuild and rerun native GGChem from the downloaded pinned ZIP")
    parser.add_argument("--extended", action="store_true", help="Use all 27 elements in the pinned Asplund2021 file")
    parser.add_argument("--check-unmodified", action="store_true", help="Also compare the uninstrumented native output")
    args = parser.parse_args()
    elements = ["H", "He"] + [e for e in abundances() if e not in ("H", "He")] if args.extended else "H He C N O P S".split()
    prefix = "native27" if args.extended else "native"
    binary = prepare_binary() if args.run else UPSTREAM / "src16/ggchem16"
    cases, checks = {}, {}
    for key in ("janaf", "burcat"):
        run = OUTPUT / f"{prefix}_{key}"
        if args.run:
            run_native(binary, run, elements, key == "burcat")
        cases[key] = read_case(run)
        if args.check_unmodified:
            original = OUTPUT / f"{prefix}_{key}_original"
            run_native(UPSTREAM / "ggchem16_original", original, elements, key == "burcat")
            checks[key] = (run / "Static_Conc.dat").read_bytes() == (original / "Static_Conc.dat").read_bytes()
            if not checks[key]:
                raise RuntimeError(f"Output-only instrumentation changed native output: {key}")
    paths = [Path(__file__), ABUNDANCES, OUTPUT / "ggchem_upstream.zip"]
    paths += [UPSTREAM / "data" / name for name in BASE_FILES + ["dispol_BURCAT.dat"]]
    with zipfile.ZipFile(OUTPUT / "ggchem_upstream.zip") as archive:
        upstream_hashes = {name.split(f"GGchem-{REVISION}/", 1)[1]: hashlib.sha256(archive.read(name)).hexdigest()
                           for name in archive.namelist() if "/src16/" in name and
                           (name.endswith((".f", ".F", ".f90")) or name.endswith("makefile.prodimo"))}
    report = {"cases": cases, "revision": REVISION,
              "scope": "Native mixed-source database selections at declared conditions, not Ohno's unavailable original inputs. Original fit validity intervals are not retained in dispol files.",
              "data_priority": {"janaf": BASE_FILES, "burcat": BASE_FILES + ["dispol_BURCAT.dat"]},
              "network": "27 solar elements" if args.extended else "HHeCNOPS",
              "instrumentation": "Only diagnostic writes of final lnK after TiC and primitive log densities; all temperatures use SMCHEM16.",
              "filters": {"charge": False, "condensation": False, "Natmax": 16},
              "upstream_source_sha256": upstream_hashes,
              "uninstrumented_output_byte_identical": checks,
              "selected_file_sha256": {str(p.relative_to(ROOT)): digest(p) for p in paths}}
    (OUTPUT / f"{prefix}_cases.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {k: v for k, v in report.items() if k != "cases"}
    manifest["cases"] = {}
    for key, case in cases.items():
        records = case["effective_records"]
        manifest["cases"][key] = {
            "elements": case["elements"], "species_count": len(case["species"]),
            "molecule_count": len(records), "temperature_count": len(case["temperature_K"]),
            "pressure_bar": case["pressure_bar"], "temperature_range_K": [200., 3000.],
            "P_molecules": {s: {k: v for k, v in record.items() if k != "coefficients"}
                            for s, record in records.items() if record["composition"].get("P", 0)},
            "selected_source_counts": dict(Counter(record["source"] for record in records.values())),
            "name_collisions": case["name_collisions"], "closure": case["native_closure"],
            "input_sha256": case["input_sha256"],
        }
    manifest["cases_artifact_sha256"] = digest(OUTPUT / f"{prefix}_cases.json")
    (OUTPUT / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {prefix}_cases.json: " + ", ".join(f"{k}={len(c['species'])} species" for k, c in cases.items()))


if __name__ == "__main__":
    main()

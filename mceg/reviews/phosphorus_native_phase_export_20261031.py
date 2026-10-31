"""Export native GGChem phase functions at unit atomic partial pressures.

The unchanged SUPERSAT and CHECK_MELTING routines evaluate functions, not
equilibrium compositions. Only the demonstration driver is replaced in an
isolated build. Native NH4SH uses a square-root saturation convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REVISION = "65bbb8f9ca3d6c0b4e58cb7dd3862322452d0d4b"
ARCHIVE_SHA256 = "4eb41336b01919f101496473c15e1d31ff9242422e220f00910d6aa73e0044cd"
DRIVER = """      SUBROUTINE DEMO_SWEEP
      use CHEMISTRY,ONLY: NMOLE
      use DUST_DATA,ONLY: NELEM,NDUST,bk,bar,dust_nam
      implicit none
      integer,parameter :: qp=selected_real_kind(33,4931)
      real*8 :: T
      real(kind=qp) :: na(NELEM),nm(NMOLE),lk(NMOLE),ss(NDUST)
      real(kind=qp) :: hc(NDUST)
      integer :: i,ios
      open(94,file='unit_atom_input.dat',status='old')
      open(95,file='unit_atom_phases.dat',status='replace')
      do
        read(94,*,iostat=ios) T,lk
        if (ios/=0) exit
        na=bar/(bk*T)
        nm=bar/(bk*T)*EXP(lk)
        call SUPERSAT(T,na,nm,ss)
        hc=-LOG(ss)
        do i=1,NDUST
          if (dust_nam(i)=='NH4SH[s]') hc(i)=2*hc(i)
        enddo
        write(95,'(*(ES27.18E4,1X))') T,hc
      enddo
      close(94)
      close(95)
      end
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "results/phosphorus_cold_validation/native7_phase_inputs.json")
    parser.add_argument("--build-dir", type=Path,
                        default=ROOT / "results/phosphorus_cold_validation/native_phase_function_build")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    archive = ROOT / "results/ohno_p3/ggchem_upstream.zip"
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Pinned native archive hash mismatch")
    source = args.build_dir / "src16"
    source.mkdir(parents=True, exist_ok=True)
    hashes = {}
    upstream = ROOT / "results/ohno_p3" / f"GGchem-{REVISION}"
    with zipfile.ZipFile(archive) as zipped:
        dust_raw = zipped.read(f"GGchem-{REVISION}/data/DustChem.dat")
        if (upstream / "data/DustChem.dat").read_bytes() != dust_raw:
            raise ValueError("Native dust data differ from the pinned archive")
        dust_hash = hashlib.sha256(dust_raw).hexdigest()
        prefix = f"GGchem-{REVISION}/src16/"
        for name in zipped.namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            relative = name[len(prefix):]
            if "/" in relative:
                continue
            raw = zipped.read(name)
            (source / relative).write_bytes(raw)
            hashes[relative] = hashlib.sha256(raw).hexdigest()
    (source / "demo_sweep.f").write_text(DRIVER)
    with (args.build_dir / "build.log").open("w") as log:
        subprocess.run(["make", "-f", "makefile.prodimo", "-j1"], cwd=source,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    run = args.build_dir / "run"
    run.mkdir(exist_ok=True)
    if not (run / "data").exists():
        (run / "data").symlink_to(upstream / "data", target_is_directory=True)
    original_run = ROOT / "results/phosphorus_cold_validation/native7_condensed"
    for filename in ("input.in", "abundances.dat"):
        (run / filename).write_bytes((original_run / filename).read_bytes())
    atoms = len(data["elements"])
    rows = np.column_stack((data["temperature_K"], np.asarray(data["gas_lnK_bar"])[:, atoms:]))
    np.savetxt(run / "unit_atom_input.dat", rows, fmt="%.18e")
    with (run / "run.log").open("w") as log:
        subprocess.run([str((source / "ggchem16").resolve()), "input.in"], cwd=run,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=60)
    values = np.loadtxt(run / "unit_atom_phases.dat")
    if values.shape != (len(rows), len(data["condensate_species"])+1):
        raise ValueError("Native phase export has incorrect shape")
    np.testing.assert_allclose(values[:, 0], rows[:, 0], rtol=0, atol=1e-12)
    if (run / "dispol.out").read_bytes() != (original_run / "dispol.out").read_bytes():
        raise ValueError("Native gas source selection changed in the function export")
    previous = np.asarray(data["condensate_hvector_RT"])
    difference = float(np.max(abs(previous-values[:, 1:])))
    if not np.isfinite(values).all() or difference > 1e-7:
        raise ValueError(f"Direct export disagrees with state reconstruction: {difference}")
    data["condensate_hvector_RT"] = values[:, 1:].tolist()
    data["scope"] = ("Matching native thermochemical functions at 31 exact temperature knots. "
                     "Phase g/RT is evaluated directly by unchanged SUPERSAT at 1 bar atomic partial pressures "
                     "with compatible molecular formation constants and native CHECK_MELTING corrections. "
                     "No native equilibrium composition enters this function evaluation or an ExoGibbs seed.")
    receipt = {
        "method": "Unit atomic partial pressures; molecular p/bar=exp(native lnK_bar). Phase h/RT=-lnS except NH4SH=-2lnS.",
        "native_equilibrium_solver_called": False,
        "native_melting_corrections_enabled": True,
        "maximum_abs_difference_from_condensed_state_reconstruction_RT": difference,
        "comparison_threshold_RT": 1e-7,
        "source_revision": REVISION,
        "source_archive_sha256": ARCHIVE_SHA256,
        "dust_data_sha256": dust_hash,
        "original_source_sha256": {name: hashes[name] for name in
                                   ("supersat.f", "init_dustchem.f", "init_chemistry.f", "read_parameter.f")},
        "driver_sha256": digest(source / "demo_sweep.f"),
        "script_sha256": digest(Path(__file__)),
        "binary_sha256": digest(source / "ggchem16"),
        "native_input_sha256": {name: digest(run / name) for name in
                                ("input.in", "abundances.dat", "unit_atom_input.dat", "unit_atom_phases.dat")}}
    data["direct_phase_function_export"] = receipt
    args.input.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    (args.build_dir / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("Exported", len(values), "native phase states; maximum reconstruction difference", difference)


if __name__ == "__main__":
    main()

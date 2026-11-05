"""Construct the exact HO2 NASA thermal + balanced ATcT candidate offline."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np

import ho2_source_review as source
from tio_candidate import _nasa

ROOT = source.ROOT
DATA = ROOT / "tests/data/validation/ho2_candidate/20261105"
R, NASA_R = source.R, source.NASA_R
COMPOSITION = {"H": 1, "O": 2}
INPUTS = (source.DATA / "nasa_records.json", source.DATA / "atct_nodes.json",
          source.DATA / "sources.json", source.DATA / "H-043.txt",
          ROOT / "tests/data/validation/oh_source_review/20261105/H-001.txt",
          ROOT / "tests/data/validation/oh_source_review/20261105/O-001.txt")
PROPERTY_KEYS = ("G_relative_J_mol", "H_relative_J_mol", "S_J_mol_K", "Cp_J_mol_K")


@lru_cache(maxsize=1)
def load_candidate() -> dict:
    """Read the frozen exact reaction coefficients and source conventions."""
    return json.loads((DATA / "candidate.json").read_text())


def _check_temperature(temperature_K: float) -> float:
    if not math.isfinite(temperature_K) or not 1500 <= temperature_K <= 4000:
        raise ValueError("Temperature outside HO2 candidate domain")
    return float(temperature_K)


def molecular_properties(temperature_K: float) -> dict:
    """Return ground-relative molecular G/H (J/mol), S/Cp (J/mol/K)."""
    t = _check_temperature(temperature_K)
    g, first, second = _nasa(t, load_candidate()["nasa_molecular_thermal_coefficients"], np)
    return dict(zip(PROPERTY_KEYS, map(float, (R*g, R*(g-t*first), -R*first, -R*t*second))))


def formation_h(temperature_K: float) -> float:
    """Return Gformation/(RT) = -ln K for H + 2 O -> HO2, at one bar."""
    t = _check_temperature(temperature_K)
    return float(_nasa(t, load_candidate()["formation_nasa9_coefficients"], np)[0]/t)


def formation_h_jax(temperature_K):
    """Support traced scalar/array inputs; return NaN outside 1500--4000 K."""
    import jax.numpy as jnp

    coefficients = load_candidate()["formation_nasa9_coefficients"]
    t = jnp.asarray(temperature_K, dtype=jnp.asarray(coefficients).dtype)
    value = _nasa(t, coefficients, jnp)[0]/t
    return jnp.where(jnp.isfinite(t) & (t >= 1500) & (t <= 4000), value, jnp.nan)


def janaf_formation_h_node(temperature_K: float) -> float:
    """Evaluate only an original JANAF knot with the same energy/NASA atoms.

    This is a pointwise thermal control, never a continuous interpolation.
    The balanced ATcT energy is inserted once, replacing native JANAF D0.
    """
    t = _check_temperature(temperature_K)
    tables = {s: source.read_table(path) for s, path in zip(("HO2", "H", "O"), INPUTS[3:])}
    if not all(t in table for table in tables.values()):
        raise ValueError("Temperature is not an archived JANAF knot")
    records = json.loads(INPUTS[0].read_text())
    comparison = source.source_comparison(tables, records, t)
    ln_k = comparison["ln_K"] + (load_candidate()["D0_J_mol"]
        - 1000*comparison["native_source_D0_kJ_mol"])/(R*t)
    ln_k -= comparison["source_minus_nasa_ln_K_components"]["atomic_thermal"]
    return -float(ln_k)


def build() -> dict:
    """Balance the same-version ATcT values and exact NASA9 reaction."""
    records = json.loads(INPUTS[0].read_text())
    atct = json.loads(INPUTS[1].read_text())
    assert atct["version"] == "1.172"
    h0 = {s: record["hf298_J_mol"]-record["h298_minus_h0_J_mol"] for s, record in records.items()}
    coefficients = {s: np.array(next(row["coefficients"] for row in record["intervals"]
                    if row["bounds_K"] == [1000., 6000.])) for s, record in records.items()}
    d0 = 1000*(sum(n*atct["records"][s]["hf0_kJ_mol"] for s,n in COMPOSITION.items())
               -atct["records"]["HO2"]["hf0_kJ_mol"])
    nasa_d0 = sum(n*h0[s] for s,n in COMPOSITION.items())-h0["HO2"]
    shift = d0/R-nasa_d0/NASA_R
    reaction = coefficients["HO2"]-sum(n*coefficients[s] for s,n in COMPOSITION.items())
    reaction[7] -= shift
    molecule = coefficients["HO2"].copy()
    molecule[7] -= h0["HO2"]/NASA_R
    return {"id": "ho2_nasa_thermal_atct_1_172_candidate_v1", "species": "H1O2",
            "reaction": "H(g)+2O(g)->HO2(g)", "temperature_range_K": [1500,4000],
            "standard_pressure_Pa": 100000, "source_interval_K": [1000,6000],
            "representation": "Exact NASA9 molecular-minus-atoms reaction, constant b1 energy shift; no fit",
            "D0_J_mol": d0, "D0_over_k_K": d0/R, "nasa_D0_J_mol": nasa_d0,
            "ATcT_minus_nasa_D0_over_k_K": shift,
            "gas_constant_J_mol_K": R, "nasa_native_gas_constant_J_mol_K": NASA_R,
            "formation_nasa9_coefficients": reaction.tolist(),
            "nasa_molecular_thermal_coefficients": molecule.tolist(),
            "thermal_convention": "NASA dimensionless thermal functions retained; ground H0 removed using native NASA R; calorics use modern SI R; same-version ATcT D0 inserted once",
            "janaf_control": "Original source nodes only, aligned to identical ATcT D0 and NASA atomic thermal functions; not a fitted continuous alternative",
            "uncertainty_scope": "No reaction covariance or high-temperature electronic-state completeness bound; inherited source inspection limits retained",
            "source_sha256": {str(path.relative_to(ROOT)): source.file_hash(path) for path in INPUTS}}


def validate() -> dict:
    """Check the continuous construction and caloric derivatives independently."""
    import jax

    model = load_candidate()
    for name, expected in model["source_sha256"].items():
        if source.file_hash(ROOT/name) != expected:
            raise ValueError("Changed HO2 source input: "+name)
    records = json.loads(INPUTS[0].read_text())
    h0_ho2 = records["HO2"]["hf298_J_mol"]-records["HO2"]["h298_minus_h0_J_mol"]
    temperatures = np.arange(1500.,4001.)
    expected, expected_h, expected_s, expected_cp, molecular = [],[],[],[],[]
    for t in temperatures:
        raw = {s: np.array(source.nasa_properties(record,t,NASA_R)) for s,record in records.items()}
        reaction = raw["HO2"]-sum(n*raw[s] for s,n in COMPOSITION.items())
        expected.append((reaction[0]/t-reaction[1])/NASA_R-model["ATcT_minus_nasa_D0_over_k_K"]/t)
        expected_h.append(R*(reaction[0]/NASA_R-model["ATcT_minus_nasa_D0_over_k_K"]))
        expected_s.append(R*reaction[1]/NASA_R)
        expected_cp.append(R*reaction[2]/NASA_R)
        h,s,cp = raw["HO2"]
        h,s,cp = (h-h0_ho2)*R/NASA_R,s*R/NASA_R,cp*R/NASA_R
        molecular.append([h-t*s,h,s,cp])
    actual = np.array([formation_h(t) for t in temperatures])
    properties = np.array([[molecular_properties(t)[key] for key in PROPERTY_KEYS] for t in temperatures])
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64",True)
    try:
        value = np.asarray(jax.jit(formation_h_jax)(temperatures))
        first = np.asarray(jax.jit(jax.vmap(jax.grad(formation_h_jax)))(temperatures))
        second = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(formation_h_jax))))(temperatures))
    finally:
        jax.config.update("jax_enable_x64",previous)
    samples=[]
    for t in np.arange(1500.,4001.,500.):
        janaf = janaf_formation_h_node(t)
        samples.append({"temperature_K":float(t),"nasa_ATcT_over_current_K":math.exp(-formation_h(t)-source.curated_properties(t)["ln_K"]),
                        "janaf_same_energy_atoms_over_nasa_K":math.exp(formation_h(t)-janaf),
                        "janaf_control_formation_h":janaf})
    return {"temperature_count":len(temperatures),"temperature_range_K":[1500,4000],"step_K":1,
            "maximum_absolute_formation_h_reconstruction_error":float(np.max(abs(actual-expected))),
            "maximum_absolute_molecular_property_reconstruction_errors":dict(zip(PROPERTY_KEYS,np.max(abs(properties-molecular),axis=0).tolist())),
            "jax_float64_maximum_absolute_errors":{
                "formation_h":float(np.max(abs(value-expected))),
                "reaction_H_J_mol":float(np.max(abs(-R*temperatures**2*first-expected_h))),
                "reaction_S_J_mol_K":float(np.max(abs(-R*(value+temperatures*first)-expected_s))),
                "reaction_Cp_J_mol_K":float(np.max(abs(-R*(2*temperatures*first+temperatures**2*second)-expected_cp)))},
            "thermal_comparison":samples,"production_change":"none",
            "uncertainty_scope":model["uncertainty_scope"]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.build:
        DATA.mkdir(parents=True,exist_ok=True)
        (DATA/"candidate.json").write_text(json.dumps(build(),indent=2,allow_nan=False)+"\n")
    result=json.dumps(validate(),indent=2,allow_nan=False)+"\n"
    if args.output: args.output.write_text(result)
    else: print(result,end="")
    if args.build and args.output:
        receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"command":" ".join(sys.argv),"result":"passed",
                 "selected_file_sha256":{str(path.relative_to(ROOT)):source.file_hash(path) for path in (*INPUTS,Path(__file__),ROOT/"mceg/reviews/ho2_source_review.py",ROOT/"mceg/reviews/tio_candidate.py",ROOT/"mceg/reviews/c2h_source_review.py",ROOT/"mceg/reviews/gas_source_comparison.py")},
                 "output_sha256":{path.name:source.file_hash(path) for path in (DATA/"candidate.json",args.output)}}
        (DATA/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")

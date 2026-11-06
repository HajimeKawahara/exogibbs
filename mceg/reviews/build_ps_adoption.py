"""Package a compact, derivative-verified PS POPS thermal update."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
from numpy.polynomial.chebyshev import chebder, chebval

import ps_candidate as candidate

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "tests/data/validation/ps_candidate/20261106/candidate.json"
BASE = ROOT / "src/exogibbs/data/curated/cn_v1.json"
OUTPUT = ROOT / "src/exogibbs/data/curated/ps_v1.json"
DATA = ROOT / "tests/data/validation/curated_ps_adoption/20261106"
BASE_SHA256 = "28dc909f1d71e0618812be34c98246d14175156b161d7eb2b7e5dbafaa1baf47"
CANDIDATE_SHA256 = "b006a86246de1813b837bf864459649b84e004b73619f33d6d708e6c941b9764"
DEGREE, FIT_NODES = 18, 80
LOWER, UPPER = 1500., 4000.
CENTER = (math.log(LOWER)+math.log(UPPER))/2
HALF_WIDTH = (math.log(UPPER)-math.log(LOWER))/2
R = candidate.R
FIT_LIMITS = {"ln_Q":1e-12, "G_J_mol":5e-8, "H_J_mol":1e-6,
              "S_J_mol_K":1e-9, "Cp_J_mol_K":1e-7}


def digest(path: Path) -> str:
    """Hash the actual construction input or output bytes."""
    return sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def level_data() -> tuple:
    """Load only the reviewed fixed-cutoff levels; never reevaluate the threshold."""
    if digest(CANDIDATE) != CANDIDATE_SHA256:
        raise ValueError("PS adoption requires the reviewed candidate bytes")
    model = candidate.load_candidate()
    for path, expected in model["source_sha256"].items():
        if digest(ROOT/path) != expected:
            raise ValueError(f"PS adoption input changed: {path}")
    levels = candidate.states()
    assert len(levels) == 223645
    return levels[:,0]*candidate.C2, levels[:,1]/model["nuclear_spin_divisor"]


def direct_log_moments(temperature_K: float) -> np.ndarray:
    """Return lnQ and its first/second derivatives with respect to lnT."""
    energy, statistical_weight = level_data()
    t = float(temperature_K)
    weights = statistical_weight*np.exp(-energy/t)
    q = float(np.sum(weights))
    mean = float(np.sum(weights*energy)/q)
    variance = float(np.sum(weights*(energy-mean)**2)/q)
    return np.array([math.log(q),mean/t,variance/t**2-mean/t])


@lru_cache(maxsize=1)
def fit_coefficients() -> tuple:
    """Project lnQ-2lnT at 80 Chebyshev nodes with compensated scalar sums."""
    angle = np.pi*(np.arange(FIT_NODES)+.5)/FIT_NODES
    temperatures = np.exp(CENTER+HALF_WIDTH*np.cos(angle))
    values = np.array([direct_log_moments(t)[0]-2*math.log(t) for t in temperatures])
    coefficients = np.array([2/FIT_NODES*math.fsum(values*np.cos(k*angle))
                             for k in range(25)])
    coefficients[0] /= 2
    return tuple(coefficients)


def build_data() -> dict:
    """Extend the immutable CN base with a compact version of the reviewed PS sum."""
    if digest(BASE) != BASE_SHA256:
        raise ValueError("PS adoption requires its pinned CN base")
    model = candidate.load_candidate()
    coefficients = fit_coefficients()[:DEGREE+1]
    translation_constant = (1.5*math.log(2*math.pi*model["molecular_mass_u"]*candidate.AMU*candidate.KB/candidate.PLANCK**2)
                            +math.log(candidate.KB/model["standard_pressure_Pa"]))
    return {"id":"curated-ps-v1","species":"P1S1","reaction":"P(g) + S(g) -> PS(g)",
            "base_dataset":"curated-cn-v1","base_data_sha256":BASE_SHA256,
            **{key:model[key] for key in ("temperature_range_K","standard_pressure_Pa",
                "gas_constant_J_mol_K","NASA_native_R_J_mol_K","original_source_D0_J_mol",
                "delta_D0_J_mol","D0_evaluation_margin_J_mol","bound_cutoff_cm-1",
                "NASA_molecular_coefficients","NASA_molecular_H0_gauge_J_mol",
                "molecular_mass_u","isotope_mass_u","nuclear_spin_divisor","symmetry_number")},
            "log_partition_representation":{"basis":"Chebyshev", "degree":DEGREE,
                "coordinate":"(ln(T)-ln_temperature_center)/ln_temperature_half_width",
                "ln_temperature_center":CENTER,"ln_temperature_half_width":HALF_WIDTH,
                "leading_ln_T_coefficient":2.,"coefficients":list(coefficients)},
            "translation_log_constant":translation_constant,
            "retained_levels":223645,"distributed_levels":225997,
            "decision":{"name":"PS","status":"prefer_conditionally","source_group":"molecular",
                "thermal_source":"POPS2017 resolved states, fixed central-D0 cutoff; NASA/Gurvich molecular thermal subtraction",
                "energy_source":"Retained Lodders2004 D0=438 +/-10 kJ/mol",
                "reason":"Adopt the authorized scoped POPS update for explicit ground/excited-state calorics, preserving current energy, atomic functions and fit/source residual. The a4Pi component remains unrefined ab initio; this is not a decisive accuracy or new precision-standard claim.",
                "condition":"1500-4000 K and one-bar standard pressure. 31P32S internal levels with retained natural-mass translation; incomplete excited-state, isotope and planetary uncertainty."},
            "sources":{"pops":"https://doi.org/10.1093/mnras/stx2229",
                "lodders":"https://doi.org/10.1063/1.1611178",
                "candidate":str(CANDIDATE.relative_to(ROOT)),
                "review":"documents/mceg/ps_candidate.rst",
                "adoption":"documents/mceg/ps_curated_adoption.rst",
                "substitution":"tests/data/validation/ps_equilibrium_impact/20261106/summary.json"},
            "source_sha256":{str(CANDIDATE.relative_to(ROOT)):CANDIDATE_SHA256,**model["source_sha256"]},
            "representation":"Add POPS-minus-NASA/Gurvich molecular Gthermal/(RT) to the inherited P1S1 fit. lnQ=2lnT plus a degree18 Chebyshev polynomial in scaled lnT. No energy shift, new isotope weights or fitted potential De adoption.",
            "interpretation":"PS-only thermal adoption on curated-cn-v1. The current Lodders energy and source-fit residual survive. Compact approximation is independently checked against direct state moments; numerical agreement does not bound physical a4Pi/manifold or atmospheric errors."}


def _fitted_log_moments(temperatures, coefficients):
    x = (np.log(temperatures)-CENTER)/HALF_WIDTH
    return np.column_stack((chebval(x,coefficients)+2*np.log(temperatures),
                            chebval(x,chebder(coefficients))/HALF_WIDTH+2,
                            chebval(x,chebder(coefficients,2))/HALF_WIDTH**2))


def _errors(temperatures, actual, expected):
    delta = actual-expected
    return {"ln_Q":float(abs(delta[:,0]).max()),
            "G_J_mol":float(abs(-R*temperatures*delta[:,0]).max()),
            "H_J_mol":float(abs(R*temperatures*delta[:,1]).max()),
            "S_J_mol_K":float(abs(R*(delta[:,0]+delta[:,1])).max()),
            "Cp_J_mol_K":float(abs(R*(delta[:,1]+delta[:,2])).max())}


def validate() -> dict:
    """Check degree convergence, independent moments and the packaged JAX function."""
    import jax
    from exogibbs.presets import curated_cn, curated_ps
    model = json.loads(OUTPUT.read_text())
    rebuilt = build_data()
    actual_coeff = np.array(model["log_partition_representation"]["coefficients"])
    rebuilt_coeff = rebuilt["log_partition_representation"].pop("coefficients")
    copy_model = json.loads(json.dumps(model));copy_model["log_partition_representation"].pop("coefficients")
    assert copy_model == rebuilt
    np.testing.assert_allclose(actual_coeff,rebuilt_coeff,atol=2e-14,rtol=0)
    ts = np.unique(np.r_[np.arange(1500.,4001.),np.random.default_rng(311367).uniform(1500,4000,257)])
    direct = np.array([direct_log_moments(t) for t in ts])
    convergence = [{"degree":degree,"maximum_absolute_errors":_errors(ts,_fitted_log_moments(ts,fit_coefficients()[:degree+1]),direct)}
                   for degree in (10,12,14,16,18,20,22,24)]
    errors = _errors(ts,_fitted_log_moments(ts,actual_coeff),direct)
    for key,limit in FIT_LIMITS.items():
        assert errors[key] < limit,(key,errors[key],limit)
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64",True)
    try:
        setup,base = curated_ps.chemsetup(),curated_cn.chemsetup()
        index = setup.species.index("P1S1")
        base_values = np.asarray(base.hvector_func(ts))
        values = np.asarray(setup.hvector_func(ts))
        np.testing.assert_array_equal(np.delete(values-base_values,index,axis=-1),0)
        f = lambda t:R*t*setup.hvector_func(t)[index]
        dg = np.asarray(jax.jit(jax.vmap(jax.grad(f)))(ts))
        ddg = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(f))))(ts))
        g = R*ts*values[:,index]
        runtime = np.column_stack((g,g-ts*dg,-dg,-ts*ddg))
        # Independent original-candidate moment formulas include the inherited fit
        # and native-NASA molecular subtraction, not fitted derivatives.
        expected = np.array([list(candidate.reaction_properties(t).values()) for t in ts])
        property_errors = dict(zip(("G_J_mol","H_J_mol","S_J_mol_K","Cp_J_mol_K"),map(float,abs(runtime-expected).max(axis=0))))
        for key,limit in FIT_LIMITS.items():
            if key != "ln_Q": assert property_errors[key] < limit,(key,property_errors[key])
        h_error = float(abs(values[:,index]-expected[:,0]/(R*ts)).max())
        assert h_error < 2e-12
        probes = [1500.,1831.25,2000.,2500.,3000.,3679.125,4000.]
        return {"dataset":model["id"],"base_dataset":model["base_dataset"],"degree":DEGREE,
                "fit_nodes":FIT_NODES,"dense_points":2501,"off_grid_points":257,"validation_points":len(ts),
                "domain_K":[1500,4000],"retained_levels":223645,"degree_convergence":convergence,
                "maximum_absolute_fit_errors":errors,"acceptance_limits":FIT_LIMITS,
                "maximum_absolute_runtime_property_errors":property_errors,
                "maximum_absolute_runtime_h_error":h_error,
                "reference_probes":[{"temperature_K":t,"formation_h":candidate.formation_h(t),
                    "reaction_properties":candidate.reaction_properties(t)} for t in probes],
                "all_other_potentials_unchanged":True,"delta_D0_J_mol":0,
                "scope":"Compact mathematical approximation of the reviewed fixed-cutoff sum; numerical fit errors are not physical uncertainty or extra energy alignment."}
    finally:
        jax.config.update("jax_enable_x64",previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--validate",action="store_true")
    args = parser.parse_args()
    if args.validate:
        report = validate()
        DATA.mkdir(parents=True,exist_ok=True)
        (DATA/"fit_validation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        paths = {Path(__file__).resolve(),CANDIDATE,BASE,OUTPUT,ROOT/"src/exogibbs/presets/curated_ps.py"}
        paths.update(ROOT/p for p in candidate.load_candidate()["source_sha256"])
        paths.update(Path(module.__file__).resolve() for module in list(sys.modules.values())
                     if getattr(module,"__file__",None) and Path(module.__file__).resolve().is_relative_to(ROOT)
                     and Path(module.__file__).suffix==".py" and Path(module.__file__).name!="exogibbs_version.py")
        paths.update((ROOT/"src/exogibbs/data/curated").glob("*_v1.json"))
        receipt={"created_utc":datetime.now(timezone.utc).isoformat(),"command":[sys.executable,*sys.argv],
                 "python":platform.python_version(),"numpy":np.__version__,"dtype":"float64","result":"passed",
                 "selected_file_sha256":{str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)},
                 "output_sha256":{"fit_validation.json":digest(DATA/"fit_validation.json")}}
        (DATA/"fit_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    else:
        args.output.write_text(json.dumps(build_data(),indent=2,allow_nan=False)+"\n")
        print(digest(args.output))

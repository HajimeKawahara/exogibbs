"""Replay inspected source expressions offline; no experimental validation.

Run with PYTHONPATH=src and JAX_PLATFORMS=cpu. Source locations, access gaps,
and retrieved-file hashes are recorded in documents/mceg/volatile_sources.rst
and the contracts. PDFs and downloaded upstream code are not required here.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform

import jax
import numpy as np

from exogibbs.solubility import (
    ch4_ardia2013,
    co2_lichtenberg2021,
    co_yoshioka2019,
    h2o_lichtenberg2021,
    h2o_sun_yao2026,
    n2_dasgupta2022,
)


ROOT = Path(__file__).resolve().parents[2]
RTOL = 2e-12


def comparisons() -> list:
    """Compare selected positive samples, retaining every mismatch as data."""
    if not jax.config.x64_enabled:
        raise RuntimeError("Set JAX_ENABLE_X64=1 for this source comparison")
    rows = []

    def add(model, reference, inputs, unit, expected, actual, purpose):
        actual = float(actual)
        rows.append({
            "model": model, "reference": reference, "purpose": purpose,
            "inputs": inputs, "output_basis": unit,
            "reference_value": expected, "implementation_value": actual,
            "implementation_over_reference": actual / expected,
            "agrees": math.isclose(actual, expected, rel_tol=RTOL, abs_tol=0.),
        })

    # Eq. 9 inverted using Table 1's ppmw coefficients, then ppmw -> fraction.
    for model, fn, alpha, beta in [
        ("h2o_lichtenberg2021", h2o_lichtenberg2021, 1.033, 1.747),
        ("co2_lichtenberg2021", co2_lichtenberg2021, 1.937e-9, .714),
    ]:
        for p in (1e5, 1e6, 1e7, 1e8):
            expected = alpha * math.exp(math.log(p) / beta) / 1e6
            add(model, "lichtenberg2021", {"partial_pressure_pa": p},
                "species_mass_fraction", expected, fn(p), "adopted_expression")

    for f, p in ((.1, .7), (.3, 1.5), (1., 3.)):
        inputs = {"methane_fugacity_gpa": f, "melt_pressure_gpa": p}
        actual = ch4_ardia2013(f, p)
        add("ch4_ardia2013", "seo2024", inputs, "CH4_mole_fraction",
            math.exp(math.log(f) - 7.63 - 1.9*p), actual, "adopted_expression")
        # A4 labels fugacity in bar while retaining the same coefficient.
        add("ch4_ardia2013", "meltyq2026_A4", inputs, "CH4_mole_fraction",
            f*1e4 * math.exp(-7.63 - 1.9*p), actual, "alternative_expression")

    for f in (1e3, 1e4, 1e5):
        inputs = {"carbon_monoxide_fugacity_bar": f}
        actual = co_yoshioka2019(f)
        # Published abstract: wt% elemental C. Its fugacity unit is not stated;
        # retain the provider's bar convention, pending the published full text.
        add("co_yoshioka2019", "yoshioka2019_abstract", inputs,
            "elemental_C_mass_fraction", 10**(-5.20 + .80*math.log10(f))/100,
            actual, "adopted_expression")
        # A5 additionally calls this CO mass fraction: compare numeric values
        # only, without claiming the species bases are physically equivalent.
        add("co_yoshioka2019", "meltyq2026_A5", inputs,
            "numeric_fraction_with_conflicting_species_label",
            10**(-7.2 + .8*math.log10(f/1e4)), actual, "alternative_expression")
    for f in (1e3, 1e5):
        add("co_yoshioka2019", "yoshioka_thesis2018",
            {"carbon_monoxide_fugacity_bar": f}, "elemental_C_mass_fraction",
            10**(-5.83 + .98*math.log10(f))/100, co_yoshioka2019(f),
            "alternative_expression")

    # MELTYQ A6 has P/T; the provider has sqrt(P)/T. P=1 is a coincidence,
    # not evidence that the two pressure dependences are equivalent.
    for p in (.5, 1., 2.):
        pn, t, iw, si, al, ti = .1, 1873., -2., .56, .11, .01
        expected = (math.sqrt(pn)*math.exp(5908*p/t - 1.6*iw)
                    + pn*math.exp(4.67 + 7.11*si - 13.06*al - 120.67*ti))/1e6
        add("n2_dasgupta2022", "meltyq2026_A6", {
            "nitrogen_partial_pressure_gpa": pn, "temperature_k": t,
            "melt_pressure_gpa": p, "delta_iw": iw,
            "x_sio2": si, "x_al2o3": al, "x_tio2": ti,
        }, "elemental_N_mass_fraction", expected,
            n2_dasgupta2022(pn, t, p, iw, x_sio2=si, x_al2o3=al, x_tio2=ti),
            "alternative_expression")

    # Scalar transcription of the inspected pinned MAGEMin H2O branch (wt%).
    # Coefficients are independent of ExoGibbs' constants; Julia is not executed.
    for pw, t, p, na, al in (
        (1., 1373., 1000., .03, .1), (100., 1673., 3000., .05, .15),
        (500., 1873., 10000., .01, .08), (2000., 2173.15, 20000., .002, .025),
    ):
        molecular = math.exp(.807805*math.log(pw) - .700336*math.log(t)
                             + 2.949559e-6*p*p/t - 74.546597*math.sqrt(p)*na/t
                             + 7146.275452*na/t)
        hydroxyl = math.exp(.449387*math.log(pw) - .360194*math.log(t)
                            + (-.308259 + 2.984703*al - 1.693710*na)*p/t)
        add("h2o_sun_yao2026", "magemin_pinned", {
            "water_partial_pressure_bar": pw, "temperature_k": t,
            "melt_pressure_bar": p, "x_na2o": na, "x_al2o3": al,
        }, "H2O_equivalent_mass_fraction", (molecular + hydroxyl)/100,
            h2o_sun_yao2026(pw, t, p, x_na2o=na, x_al2o3=al),
            "adopted_expression")
    return rows


def report() -> dict:
    """Record numerical diagnostics, environment and the inspected local files."""
    rows = comparisons()
    files = [
        "src/exogibbs/solubility/volatile.py",
        "src/exogibbs/solubility/sun_yao2026.py",
        "mceg/reviews/volatile_source_review_20261021.py",
    ]
    return {
        "review_date": "2026-10-21",
        "provider_revision": "e817ec96d92b6439aabe4bdcc17c8fbe877b8e57",
        "scope": "Formula comparisons only; no laboratory data, Julia or MATLAB execution.",
        "relative_tolerance": RTOL,
        "absolute_tolerance": 0.,
        "environment": {"python": platform.python_version(), "jax": jax.__version__,
                        "numpy": np.__version__, "backend": jax.default_backend(),
                        "x64": jax.config.x64_enabled},
        "selected_file_sha256": {
            name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files
        },
        "counts": {"total": len(rows), "consistent": sum(r["agrees"] for r in rows),
                   "discrepant": sum(not r["agrees"] for r in rows)},
        "comparisons": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report()
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    # Known alternative-expression differences are findings, not a replay error.
    if any(not r["agrees"] for r in result["comparisons"]
           if r["purpose"] == "adopted_expression"):
        raise SystemExit("An adopted-expression comparison no longer agrees")

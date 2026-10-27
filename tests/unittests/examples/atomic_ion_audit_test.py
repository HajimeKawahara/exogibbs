"""Offline regression checks for the atomic-ion audit, without importing MCEG."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/comparisons/audit_atomic_ions.py"
SPEC = importlib.util.spec_from_file_location("atomic_ion_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AtomicIonAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(audit.REFERENCE_PATH.read_text())
        cls.rows, cls.summaries = audit.compute_audit()
        cls.by_key = {(r["species"], r["temperature_K"]): r for r in cls.rows}

    def test_coverage_keeps_unbound_and_missing_data_distinct(self):
        self.assertEqual(len(self.summaries), 49)
        self.assertEqual(sum(r["charge"] == 1 for r in self.summaries), 27)
        self.assertEqual(len(self.rows), 343)
        self.assertEqual(sum(r["log10_Kp_reference"] is not None for r in self.rows), 329)
        self.assertEqual(self.reference["nasa9_unavailable"], ["Ca1-", "Zn1-"])
        for species in ("N1-", "Zn1-"):
            for t in self.reference["temperature_K"]:
                row = self.by_key[species, t]
                self.assertEqual(row["reference_status"], "no_stable_bound_anion")
                self.assertIsNone(row["log10_Kp_reference"])
                self.assertIsNone(row["dex_fastchem_over_reference"])
        self.assertIsNotNone(self.by_key["N1-", 1000]["log10_Kp_nasa9"])
        self.assertIsNone(self.by_key["Ca1-", 1000]["log10_Kp_nasa9"])
        self.assertIsNotNone(self.by_key["Ca1-", 1000]["log10_Kp_reference"])

    def test_reproduces_previous_ni_co_audit(self):
        path = ROOT / "tests/data/validation/ni_co_ionization/comparison.csv"
        with path.open() as stream:
            for old in csv.DictReader(stream):
                new = self.by_key[old["element"] + "1+", float(old["temperature_K"])]
                for new_key, old_key in (("reference", "saha"), ("fastchem", "fastchem"),
                                         ("nasa9", "nasa9")):
                    self.assertAlmostEqual(new["log10_Kp_" + new_key],
                                           float(old["log10_Kp_" + old_key]), places=11)

    def test_hydrogen_spin_pressure_and_reaction_directions(self):
        for t in self.reference["temperature_K"]:
            # Independent NASA fits catch a missing electron spin factor or a
            # pressure-unit error in either direction of the statistical model.
            for species in ("H1+", "H1-"):
                self.assertLess(abs(self.by_key[species, t]["dex_nasa9_over_reference"]), .001)
            plus = self.by_key["H1+", t]["log10_Kp_reference"]
            minus = self.by_key["H1-", t]["log10_Kp_reference"]
            # Electron translational and spin factors cancel for 2H -> H+ + H-.
            atom = self.reference["atoms"]["H"]
            energy = atom["ionization_energy_eV"] - self.reference["anions"]["H"]["electron_affinity_eV"]
            i = self.reference["temperature_K"].index(t)
            q = atom["partition_neutral"][i]
            expected = (-2 * math.log(q) - energy / (8.617333262145179e-5 * t)) / math.log(10)
            self.assertAlmostEqual(plus + minus, expected, places=11)

    def test_copper_discrepancy_is_already_in_janaf(self):
        for t in self.reference["temperature_K"]:
            row = self.by_key["Cu1+", t]
            self.assertLess(abs(row["log10_Kp_fastchem"] - row["log10_Kp_janaf"]), .003)
            self.assertGreater(row["dex_fastchem_over_reference"], 3)
            self.assertLess(abs(row["dex_nasa9_over_reference"]), .001)

    def test_anion_levels_are_bound_and_nasa_is_not_a_blanket_improvement(self):
        for anion in self.reference["anions"].values():
            if anion["status"] != "bound_state_reference":
                continue
            self.assertEqual(anion["levels"][0][1], 0)
            for _, energy, weight, uncertainty, _ in anion["levels"]:
                self.assertGreater(weight, 0)
                self.assertGreaterEqual(uncertainty, 0)
                self.assertLess(energy / 8065.543937349211, anion["electron_affinity_eV"])
        for species, t in (("C1-", 1000), ("Ge1-", 3000)):
            row = self.by_key[species, t]
            self.assertGreater(abs(row["dex_nasa9_over_reference"]),
                               abs(row["dex_fastchem_over_reference"]))

    def test_rejects_unaudited_domains_and_wrong_electron_sign(self):
        for t in (999, 1200, 6001, float("nan")):
            with self.assertRaises(ValueError):
                audit.reference_log_kp(t, "H", 1, self.reference)
        with self.assertRaises(ValueError):
            audit.reference_log_kp(1000, "N", -1, self.reference)
        with self.assertRaises(ValueError):
            audit.reference_log_kp(1000, "H", 2, self.reference)
        with self.assertRaises(ValueError):
            audit.read_fastchem_ions("H1+ Hydrogen : H 1 e- 1 # source\n1 2 3 4 5", {"H"})

    def test_evidence_hashes_and_contract_reference(self):
        result_dir = ROOT / "tests/data/validation/atomic_ion_audit"
        receipt = json.loads((result_dir / "receipt.json").read_text())
        for name, expected in receipt["selected_file_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected,
                             "Review MCEG evidence after changing " + name)
        for name, expected in receipt["output_sha256"].items():
            self.assertEqual(hashlib.sha256((result_dir / name).read_bytes()).hexdigest(), expected)
        # Fixed-layout metadata check, not general YAML or MCEG validation.
        contract = (ROOT / "mceg/contracts/atomic_ion_reference.yaml").read_text()
        script_hash = receipt["selected_file_sha256"][str(SCRIPT.relative_to(ROOT))]
        self.assertIn("revision: sha256:" + script_hash, contract)
        self.assertIn("tests/data/validation/atomic_ion_audit/receipt.json", contract)
        self.assertIn("tests/data/validation/atomic_ion_audit/tests.json", contract)
        self.assertIn("kind: physical_validation\n  status: unknown", contract)


if __name__ == "__main__":
    unittest.main()

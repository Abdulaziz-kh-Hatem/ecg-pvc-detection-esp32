#!/usr/bin/env python3
"""
Clinical Stress Testing Suite for PVC Detection.
Empirically investigates the two critical peer-review attack points:
1. Arrhythmia Specificity Attack: Evaluating false positive rates on excluded
   arrhythmia classes (Premature Atrial Contractions / PACs, Fusion beats).
2. R-Peak Jitter Robustness Attack: Evaluating performance degradation under
   simulated automated QRS detector temporal jitter (±10 ms, ±20 ms).

Usage:
    python scripts/run_clinical_stress_tests.py
"""

import sys
import os
import json
import subprocess

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_venv_python = os.path.join(_repo_root, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    _res = subprocess.run([_venv_python, os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(_res.returncode)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

sys.path.insert(0, _repo_root)

from validation.stress_tests.clinical_stress import (
    run_arrhythmia_specificity_stress_test,
    run_rpeak_jitter_stress_test,
    plot_stress_test_figure,
)


def main() -> None:
    results_dir = os.path.join(_repo_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    arrhy_res = run_arrhythmia_specificity_stress_test(data_dir=os.path.join(_repo_root, "data", "raw"))
    jitter_res = run_rpeak_jitter_stress_test(data_dir=os.path.join(_repo_root, "data", "raw"))

    stress_test_data = {
        "arrhythmia_specificity_test": arrhy_res,
        "rpeak_jitter_robustness_test": jitter_res
    }

    out_json = os.path.join(results_dir, "clinical_stress_test_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stress_test_data, f, indent=2)
    print(f"\nSaved stress test metrics to {out_json}")

    plot_stress_test_figure(jitter_res, arrhy_res, out_dir=os.path.join(_repo_root, "figures", "validated"))
    print("=" * 80)
    print("Clinical stress testing completed successfully.")


if __name__ == "__main__":
    main()

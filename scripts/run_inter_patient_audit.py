#!/usr/bin/env python3
"""
Standalone CLI entry point for executing the inter-patient (Leave-One-Patient-Out) audit.

Usage:
    python scripts/run_inter_patient_audit.py
"""

import sys
import os
import subprocess

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_venv_python = (
    os.path.join(_repo_root, ".venv", "Scripts", "python.exe")
    if sys.platform == "win32"
    else os.path.join(_repo_root, ".venv", "bin", "python")
)
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    _res = subprocess.run([_venv_python, os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(_res.returncode)

sys.path.insert(0, _repo_root)

from validation.lopo.pipeline import run_inter_patient_lopo_pipeline


def main() -> None:
    config_path = os.path.join(_repo_root, "configs", "canonical_config.yaml")
    results_df = run_inter_patient_lopo_pipeline(config_path)
    print("\nInter-Patient audit finished successfully.")
    print("Results saved to results/inter_patient_results.csv")


if __name__ == "__main__":
    main()

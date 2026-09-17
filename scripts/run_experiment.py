#!/usr/bin/env python3
"""
PVC Detection Experiment Runner.
Primary execution entry point for the undergraduate research project:
- Subject-specific intra-patient 70/30 chronological evaluation
- 15 MIT-BIH records, causal digital filtering, median template calibration
- Dual-voting feature selection (RF + XGBoost) and balanced Decision Tree classification

Usage:
    python scripts/run_experiment.py
    # or
    python -m scripts.run_experiment
"""

import sys
import os
import subprocess

# Auto-delegate to local virtualenv if present and not already active
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_venv_python = (
    os.path.join(_repo_root, ".venv", "Scripts", "python.exe")
    if sys.platform == "win32"
    else os.path.join(_repo_root, ".venv", "bin", "python")
)
if os.path.exists(_venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(_venv_python).lower():
    _res = subprocess.run([_venv_python, os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(_res.returncode)

# Add repository root to python search path
sys.path.insert(0, _repo_root)

from src.pipeline.train import run_canonical_pipeline


def main() -> None:
    config_path = os.path.join(_repo_root, "configs", "canonical_config.yaml")
    results_df, all_features, p208_cm, p208_model = run_canonical_pipeline(config_path)

    print("\nCanonical experiment finished successfully.")
    print("Results saved to:")
    print("  - results/patient_results.csv")
    print("  - results/feature_selection.json")


if __name__ == "__main__":
    main()

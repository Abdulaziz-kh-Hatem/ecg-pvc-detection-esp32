"""
Clinical stress testing module.
Evaluates arrhythmia specificity against non-PVC ectopy (PACs, Fusion beats)
and temporal robustness against automated QRS detection jitter.
"""

from validation.stress_tests.clinical_stress import (
    run_arrhythmia_specificity_stress_test,
    run_rpeak_jitter_stress_test,
    plot_stress_test_figure,
)

__all__ = [
    "run_arrhythmia_specificity_stress_test",
    "run_rpeak_jitter_stress_test",
    "plot_stress_test_figure",
]

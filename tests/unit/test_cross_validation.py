"""
Unit tests for cross-validation protocols and statistical uncertainty calculations.
"""

import numpy as np
import pytest
from validation.cross_validation import (
    generate_lopo_splits,
    generate_grouped_kfold_splits,
    compute_bootstrap_ci
)


def test_lopo_splits_invariants():
    patient_ids = ["105", "106", "119", "200", "201"]
    splits = generate_lopo_splits(patient_ids)

    assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"

    tested_patients = set()
    for train_pids, test_pid in splits:
        # Invariant 1: Zero overlap between train and test patients
        assert test_pid not in train_pids, f"Patient {test_pid} leaked into train partition!"
        # Invariant 2: Exactly 1 test patient, N-1 train patients
        assert len(train_pids) == len(patient_ids) - 1
        assert test_pid in patient_ids
        tested_patients.add(test_pid)

    # Invariant 3: Every patient is tested exactly once
    assert tested_patients == set(patient_ids)


def test_grouped_kfold_splits_invariants():
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, size=100)
    groups = np.array(["P1"] * 20 + ["P2"] * 20 + ["P3"] * 20 + ["P4"] * 20 + ["P5"] * 20)

    splits = generate_grouped_kfold_splits(X, y, groups=groups, n_splits=5)
    assert len(splits) == 5

    for train_idx, test_idx in splits:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        # Invariant: No patient overlap across fold boundary
        assert len(train_groups.intersection(test_groups)) == 0


def test_bootstrap_ci_coverage_and_invariants():
    values = np.array([99.0, 99.5, 99.2, 98.8, 99.6, 99.4, 99.1])
    mean_val, lower, upper = compute_bootstrap_ci(values, n_bootstraps=500, ci=95.0, random_state=42)

    # Mathematical Invariant: lower <= mean <= upper
    assert lower <= mean_val <= upper, f"Bootstrap bounds violated: {lower} <= {mean_val} <= {upper}"
    assert lower > 98.0 and upper < 100.0


def test_cross_validation_error_handling():
    # Insufficient patient list should raise ValueError
    with pytest.raises(ValueError):
        generate_lopo_splits(["single_patient"])

    # Invalid confidence interval should raise ValueError
    with pytest.raises(ValueError):
        compute_bootstrap_ci(np.array([1, 2, 3]), ci=150.0)

    # Empty array should raise ValueError
    with pytest.raises(ValueError):
        compute_bootstrap_ci(np.array([]))

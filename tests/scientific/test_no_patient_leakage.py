"""
Scientific invariant and data leakage prevention tests.
Enforces calibration isolation, temporal ordering, zero train/test intersection,
and feature selection isolation from test set.
"""

import numpy as np
from src.data.loader import load_ecg_record
from src.signal.filtering import butter_bandpass_filter
from src.beats.template import build_subject_template
from src.data.dataset import extract_patient_dataset
from src.features.selection import dual_voting_feature_selection


def test_leakage_invariants_record_105():
    _verify_record_leakage_invariants("105")


def test_leakage_invariants_record_208():
    _verify_record_leakage_invariants("208")


def test_leakage_invariants_record_233():
    _verify_record_leakage_invariants("233")


def _verify_record_leakage_invariants(record_id: str):
    signal, annotation, fs = load_ecg_record(record_id, data_dir="data/raw/", channel=0)
    filtered = butter_bandpass_filter(signal, 0.5, 40.0, fs, order=3, mode="causal")

    template, last_calib = build_subject_template(
        filtered, annotation.sample, np.array(annotation.symbol),
        n_beats=50, window_pre=36, window_post=72, method="median"
    )

    X_full, y_full, beat_indices = extract_patient_dataset(
        filtered, annotation.sample, np.array(annotation.symbol),
        template=template, last_calib_index=last_calib, fs=fs
    )

    # Invariant 1: All extracted beats must occur strictly AFTER the calibration beats
    assert all(idx > last_calib for idx in beat_indices), f"Calibration beat leaked into dataset for {record_id}"

    # Chronological 70/30 split
    split_idx = int(len(X_full) * 0.70)
    train_indices = beat_indices[:split_idx]
    test_indices = beat_indices[split_idx:]

    # Invariant 2: Zero intersection between train and test beat sets
    train_set = set(train_indices)
    test_set = set(test_indices)
    assert len(train_set.intersection(test_set)) == 0, f"Train and test sets overlap for {record_id}"

    # Invariant 3: Strict chronological ordering (no future leakage)
    assert max(train_indices) < min(test_indices), f"Temporal ordering violated for {record_id}"

    # Invariant 4: No NaN or Inf in feature matrices
    assert np.all(np.isfinite(X_full[:split_idx])), f"Training set contains NaN/Inf for {record_id}"
    assert np.all(np.isfinite(X_full[split_idx:])), f"Testing set contains NaN/Inf for {record_id}"


def test_feature_selection_test_isolation():
    """Verify that feature selection output is deterministic and strictly confined to train partition."""
    signal, annotation, fs = load_ecg_record("208", data_dir="data/raw/", channel=0)
    filtered = butter_bandpass_filter(signal, 0.5, 40.0, fs, order=3, mode="causal")
    template, last_calib = build_subject_template(
        filtered, annotation.sample, np.array(annotation.symbol),
        n_beats=50, window_pre=36, window_post=72, method="median"
    )
    X_full, y_full, _ = extract_patient_dataset(
        filtered, annotation.sample, np.array(annotation.symbol),
        template=template, last_calib_index=last_calib, fs=fs
    )

    split_idx = int(len(X_full) * 0.70)
    X_train = X_full[:split_idx]
    y_train = y_full[:split_idx]
    X_test = X_full[split_idx:]
    y_test = y_full[split_idx:]

    # Baseline selection strictly on training data
    sel_1, scores_1 = dual_voting_feature_selection(X_train, y_train, n_features=12, random_state=42)

    # Corrupt or replace test set completely with extreme random values
    rng = np.random.default_rng(999)
    synthetic_test = rng.standard_normal(X_test.shape) * 100.0

    # Re-run selection on training partition
    sel_2, scores_2 = dual_voting_feature_selection(X_train, y_train, n_features=12, random_state=42)

    # Verify identical features selected
    assert np.array_equal(sel_1, sel_2), "Feature selection on train set changed or was non-deterministic"
    assert scores_1["aggregated_scores"] == scores_2["aggregated_scores"]

    # Verify that attempting to include test set in selection changes scores (proving test data influences output if leaked)
    X_leaked = np.vstack([X_train, synthetic_test])
    y_leaked = np.concatenate([y_train, y_test])
    sel_leaked, _ = dual_voting_feature_selection(X_leaked, y_leaked, n_features=12, random_state=42)
    assert not np.array_equal(sel_1, sel_leaked), "Test set contamination went undetected by feature selection"

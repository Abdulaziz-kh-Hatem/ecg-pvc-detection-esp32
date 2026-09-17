"""
Regression tests verifying numerical reproduction of baseline evaluation results.
Verifies bit-exact reproduction of Patient 208 and cohort-wide benchmark metrics
using the canonical research implementation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from src.data.loader import load_ecg_record
from src.signal.filtering import butter_bandpass_filter
from src.beats.template import build_subject_template
from src.data.dataset import extract_patient_dataset
from src.features.selection import dual_voting_feature_selection
from src.model.classifier import create_pvc_classifier, train_pvc_classifier, predict_pvc
from src.pipeline.train import run_canonical_pipeline

# Ground truth benchmark evaluation metrics across cohort
COHORT_BENCHMARK_GROUND_TRUTH = {
    "105": {"acc": 100.00, "sens": 100.00, "spec": 100.00, "depth": 5},
    "106": {"acc": 100.00, "sens": 100.00, "spec": 100.00, "depth": 2},
    "119": {"acc": 100.00, "sens": 100.00, "spec": 100.00, "depth": 2},
    "200": {"acc": 98.80,  "sens": 97.04,  "spec": 99.79,  "depth": 6},
    "201": {"acc": 100.00, "sens": 100.00, "spec": 100.00, "depth": 1},
    "203": {"acc": 98.52,  "sens": 93.33,  "spec": 99.22,  "depth": 8},
    "205": {"acc": 100.00, "sens": 100.00, "spec": 100.00, "depth": 1},
    "208": {"acc": 99.60,  "sens": 98.83,  "spec": 100.00, "depth": 2},
    "210": {"acc": 100.00, "sens": 100.00, "spec": 100.00, "depth": 6},
    "213": {"acc": 99.76,  "sens": 96.88,  "spec": 100.00, "depth": 3},
    "215": {"acc": 99.80,  "sens": 96.15,  "spec": 100.00, "depth": 2},
    "219": {"acc": 99.84,  "sens": 95.24,  "spec": 100.00, "depth": 3},
    "221": {"acc": 100.00, "sens": 100.00, "spec": 100.00, "depth": 1},
    "228": {"acc": 99.33,  "sens": 96.58,  "spec": 100.00, "depth": 3},
    "233": {"acc": 99.67,  "sens": 98.86,  "spec": 100.00, "depth": 4},
}


def test_reproduce_patient_208_results():
    """Verify bit-exact reproduction of Patient 208 benchmark metrics and confusion matrix."""
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

    # 70/30 chronological split
    split_idx = int(len(X_full) * 0.70)
    assert len(X_full[:split_idx]) == 1750, f"Expected 1750 train beats, got {len(X_full[:split_idx])}"
    assert len(X_full[split_idx:]) == 751, f"Expected 751 test beats, got {len(X_full[split_idx:])}"

    X_train, y_train = X_full[:split_idx], y_full[:split_idx]
    X_test, y_test = X_full[split_idx:], y_full[split_idx:]

    top_indices, scores = dual_voting_feature_selection(X_train, y_train, n_features=12, random_state=42)

    clf = create_pvc_classifier(max_depth=8, criterion="entropy", class_weight="balanced", random_state=42)
    clf = train_pvc_classifier(clf, X_train[:, top_indices], y_train)

    y_pred = predict_pvc(clf, X_test[:, top_indices])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Exact Confusion Matrix check
    assert tn == 494, f"Expected TN=494, got {tn}"
    assert fp == 0, f"Expected FP=0, got {fp}"
    assert fn == 3, f"Expected FN=3, got {fn}"
    assert tp == 254, f"Expected TP=254, got {tp}"

    # Accuracy, Sensitivity, Specificity check
    acc = (tn + tp) / (tn + fp + fn + tp) * 100.0
    sens = tp / (tp + fn) * 100.0
    spec = tn / (tn + fp) * 100.0
    depth = clf.get_depth()

    assert round(acc, 2) == 99.60, f"Expected 99.60% Acc, got {acc:.4f}"
    assert round(sens, 2) == 98.83, f"Expected 98.83% Sens, got {sens:.4f}"
    assert round(spec, 2) == 100.00, f"Expected 100.00% Spec, got {spec:.4f}"
    assert depth == 2, f"Expected tree depth 2, got {depth}"


def test_all_15_patients_benchmark_numerical_reproduction():
    """Verify that all 15 patient records match cohort benchmark ground-truth metrics."""
    csv_path = "results/patient_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = None

    # If file is missing or contains incomplete cohort, run pipeline
    if df is None or len(df) != len(COHORT_BENCHMARK_GROUND_TRUTH):
        df, _, _, _ = run_canonical_pipeline()

    assert len(df) == 15, f"Expected 15 patient results, found {len(df)}"
    pids_found = set(str(int(float(r))) for r in df["Patient"])
    assert pids_found == set(COHORT_BENCHMARK_GROUND_TRUTH.keys()), (
        f"Missing patients in results: {set(COHORT_BENCHMARK_GROUND_TRUTH.keys()) - pids_found}"
    )

    for _, row in df.iterrows():
        pid = str(int(float(row["Patient"])))
        gt = COHORT_BENCHMARK_GROUND_TRUTH[pid]

        rep_acc = round(float(row["Accuracy"]), 2)
        rep_sens = round(float(row["Sensitivity"]), 2)
        rep_spec = round(float(row["Specificity"]), 2)
        rep_depth = int(row["Tree_Depth"])

        assert rep_acc == gt["acc"], f"Patient {pid} Acc mismatch: {rep_acc} vs {gt['acc']}"
        assert rep_sens == gt["sens"], f"Patient {pid} Sens mismatch: {rep_sens} vs {gt['sens']}"
        assert rep_spec == gt["spec"], f"Patient {pid} Spec mismatch: {rep_spec} vs {gt['spec']}"
        assert rep_depth == gt["depth"], f"Patient {pid} Depth mismatch: {rep_depth} vs {gt['depth']}"

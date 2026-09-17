"""
Clinical Stress Testing Suite for PVC Detection.
Empirically investigates two critical evaluation aspects:
1. Arrhythmia Specificity: Evaluating false positive rates on non-target
   arrhythmia classes (Premature Atrial Contractions / PACs, Fusion beats).
2. R-Peak Jitter Robustness: Evaluating performance degradation under
   simulated automated QRS detector temporal jitter (±10 ms, ±20 ms).

Reuses canonical signal processing, segmentation, features, and model modules.
"""

import os
import json
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.data.loader import load_ecg_record
from src.data.dataset import extract_patient_dataset
from src.signal.filtering import butter_bandpass_filter
from src.beats.template import build_subject_template
from src.beats.segmentation import segment_beat
from src.features.extraction import ALL_32_FEATURE_NAMES, extract_all_32_features
from src.features.selection import dual_voting_feature_selection
from src.model.classifier import create_pvc_classifier, train_pvc_classifier, predict_pvc


def run_arrhythmia_specificity_stress_test(
    records_to_test: List[str] = ["200", "201", "208", "210", "213"],
    data_dir: str = "data/raw/"
) -> Dict[str, Any]:
    """
    Evaluate how the trained subject-specific model responds to non-N and non-V beats
    (Premature Atrial Contractions 'A', Aberrated 'a', Fusion 'F').
    """
    print("\n" + "=" * 80)
    print(" 1. ARRHYTHMIA SPECIFICITY STRESS TEST (NON-PVC ECTOPY)")
    print("=" * 80)
    print(f"{'Patient':<8} | {'Symbol':<8} | {'Count':<8} | {'Classified as PVC':<18} | {'False Alarm Rate (%)'}")
    print("-" * 80)

    results: Dict[str, Any] = {}

    for pid in records_to_test:
        signal, annotation, fs = load_ecg_record(pid, data_dir=data_dir, channel=0)
        filtered = butter_bandpass_filter(signal, 0.5, 40.0, fs, order=3, mode="causal")

        template, last_calib = build_subject_template(
            filtered, annotation.sample, np.array(annotation.symbol),
            n_beats=50, window_pre=36, window_post=72, method="median"
        )

        # Standard train partition
        X_full, y_full, _ = extract_patient_dataset(
            filtered, annotation.sample, np.array(annotation.symbol),
            template=template, last_calib_index=last_calib, fs=fs
        )
        split_idx = int(len(X_full) * 0.70)
        X_train, y_train = X_full[:split_idx], y_full[:split_idx]

        top_indices, _ = dual_voting_feature_selection(X_train, y_train, n_features=12, random_state=42)
        clf = create_pvc_classifier(max_depth=8, criterion="entropy", class_weight="balanced", random_state=42)
        clf = train_pvc_classifier(clf, X_train[:, top_indices], y_train)

        beats = annotation.sample
        syms = np.array(annotation.symbol)
        rr_intervals = np.diff(beats) / fs

        target_symbols = ["A", "a", "F", "S", "J"]
        patient_results = {}

        for sym in target_symbols:
            indices = [
                i for i in range(last_calib + 1, len(beats) - 1)
                if syms[i] == sym and segment_beat(filtered, beats[i], pre_samples=36, post_samples=72) is not None
            ]
            if len(indices) == 0:
                continue

            feats_list = []
            for i in indices:
                pre_rr = float((beats[i] - beats[i - 1]) / fs)
                post_rr = float((beats[i + 1] - beats[i]) / fs)
                local_rr = float(np.mean(rr_intervals[max(0, i - 5):i])) if i > 0 else pre_rr
                seg = segment_beat(filtered, beats[i], pre_samples=36, post_samples=72)
                f_dict = extract_all_32_features(seg, template, pre_rr, post_rr, local_rr)
                feats_list.append([f_dict[f] for f in ALL_32_FEATURE_NAMES])

            X_arrhy = np.array(feats_list)[:, top_indices]
            preds = predict_pvc(clf, X_arrhy)
            n_pvc_pred = int(np.sum(preds == 1))
            fa_rate = float(n_pvc_pred / len(indices) * 100.0)

            patient_results[sym] = {
                "count": len(indices),
                "classified_as_pvc": n_pvc_pred,
                "false_alarm_rate": fa_rate
            }

            sym_label = f"Class '{sym}'"
            print(f"{pid:<8} | {sym_label:<8} | {len(indices):<8} | {n_pvc_pred:<18} | {fa_rate:<7.2f}%")

        results[pid] = patient_results

    return results


def run_rpeak_jitter_stress_test(
    record_id: str = "208",
    jitter_ms_levels: List[int] = [0, 5, 10, 15, 20],
    n_trials: int = 10,
    data_dir: str = "data/raw/"
) -> Dict[int, Dict[str, float]]:
    """
    Simulate automated QRS detector temporal localization jitter and quantify
    diagnostic performance degradation.
    """
    print("\n" + "=" * 80)
    print(f" 2. R-PEAK TEMPORAL JITTER STRESS TEST (PATIENT {record_id})")
    print("=" * 80)
    print(f"{'Jitter (ms)':<12} | {'Mean Acc(%)':<12} | {'Mean Sens(%)':<12} | {'Mean Spec(%)':<12} | {'Sensitivity Drop (%)'}")
    print("-" * 80)

    signal, annotation, fs = load_ecg_record(record_id, data_dir=data_dir, channel=0)
    filtered = butter_bandpass_filter(signal, 0.5, 40.0, fs, order=3, mode="causal")

    template, last_calib = build_subject_template(
        filtered, annotation.sample, np.array(annotation.symbol),
        n_beats=50, window_pre=36, window_post=72, method="median"
    )

    X_full, y_full, beat_indices = extract_patient_dataset(
        filtered, annotation.sample, np.array(annotation.symbol),
        template=template, last_calib_index=last_calib, fs=fs
    )

    split_idx = int(len(X_full) * 0.70)
    X_train, y_train = X_full[:split_idx], y_full[:split_idx]
    test_indices = beat_indices[split_idx:]
    y_test = y_full[split_idx:]

    top_indices, _ = dual_voting_feature_selection(X_train, y_train, n_features=12, random_state=42)
    clf = create_pvc_classifier(max_depth=8, criterion="entropy", class_weight="balanced", random_state=42)
    clf = train_pvc_classifier(clf, X_train[:, top_indices], y_train)

    baseline_sens = None
    jitter_results = {}

    beats = np.array(annotation.sample)
    rr_intervals = np.diff(beats) / fs

    for jitter_ms in jitter_ms_levels:
        jitter_samples_max = int((jitter_ms / 1000.0) * fs)
        acc_trials = []
        sens_trials = []
        spec_trials = []

        rng = np.random.default_rng(42)

        for _ in range(n_trials):
            if jitter_samples_max > 0:
                jitter_offsets = rng.integers(-jitter_samples_max, jitter_samples_max + 1, size=len(test_indices))
            else:
                jitter_offsets = np.zeros(len(test_indices), dtype=int)

            X_test_jittered = []
            valid_y_test = []

            for idx_k, i in enumerate(test_indices):
                jittered_peak = beats[i] + jitter_offsets[idx_k]
                seg = segment_beat(filtered, jittered_peak, pre_samples=36, post_samples=72)
                if seg is None:
                    continue

                pre_rr = float((beats[i] - beats[i - 1]) / fs)
                post_rr = float((beats[i + 1] - beats[i]) / fs)
                local_rr = float(np.mean(rr_intervals[max(0, i - 5):i])) if i > 0 else pre_rr

                f_dict = extract_all_32_features(seg, template, pre_rr, post_rr, local_rr)
                X_test_jittered.append([f_dict[f] for f in ALL_32_FEATURE_NAMES])
                valid_y_test.append(y_test[idx_k])

            X_test_j = np.array(X_test_jittered)[:, top_indices]
            y_test_j = np.array(valid_y_test)

            y_pred = predict_pvc(clf, X_test_j)
            cm = confusion_matrix(y_test_j, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            acc = (tn + tp) / (tn + fp + fn + tp) * 100.0
            sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0

            acc_trials.append(acc)
            sens_trials.append(sens)
            spec_trials.append(spec)

        mean_acc = float(np.mean(acc_trials))
        mean_sens = float(np.mean(sens_trials))
        mean_spec = float(np.mean(spec_trials))

        if baseline_sens is None:
            baseline_sens = mean_sens
            drop = 0.0
        else:
            drop = baseline_sens - mean_sens

        jitter_results[jitter_ms] = {
            "mean_accuracy": mean_acc,
            "mean_sensitivity": mean_sens,
            "mean_specificity": mean_spec,
            "sensitivity_drop": drop
        }

        print(f"{jitter_ms:<12} | {mean_acc:<12.2f} | {mean_sens:<12.2f} | {mean_spec:<12.2f} | {drop:<7.2f}%")

    return jitter_results


def plot_stress_test_figure(jitter_results: Dict[Any, Any], arrhythmia_results: Dict[str, Any], out_dir: str = "figures") -> None:
    """Generate visualization figure documenting stress testing outcomes."""
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Jitter Robustness Curve
    ms_vals = sorted(list(jitter_results.keys()))
    sens_vals = [jitter_results[ms]["mean_sensitivity"] for ms in ms_vals]
    spec_vals = [jitter_results[ms]["mean_specificity"] for ms in ms_vals]

    ax1.plot(ms_vals, sens_vals, marker="o", color="#d62728", linewidth=2, label="Sensitivity (Recall)")
    ax1.plot(ms_vals, spec_vals, marker="s", color="#1f77b4", linewidth=2, label="Specificity")
    ax1.axvline(x=10, color="gray", linestyle=":", label="Typical Automated Detector Jitter (±10 ms)")
    ax1.set_xlabel("R-Peak Jitter Window (± ms)")
    ax1.set_ylabel("Diagnostic Metric (%)")
    ax1.set_title("Robustness to Automated QRS Detection Jitter (Patient 208)", fontweight="bold")
    ax1.legend(loc="lower left")
    ax1.set_ylim(80, 102)

    # Plot 2: Non-PVC False Alarm Rates
    categories = []
    fa_rates = []
    colors = []

    for pid in arrhythmia_results:
        for sym in arrhythmia_results[pid]:
            count = arrhythmia_results[pid][sym]["count"]
            if count >= 10:
                categories.append(f"P{pid} '{sym}' (N={count})")
                fa_rates.append(arrhythmia_results[pid][sym]["false_alarm_rate"])
                colors.append("#d62728" if "F" in sym else "#ff7f0e")

    if categories:
        ax2.barh(categories, fa_rates, color=colors, edgecolor="black")
        ax2.set_xlabel("False Positive Rate (% Classified as PVC)")
        ax2.set_title("Vulnerability to Non-PVC Arrhythmias (PACs & Fusion Beats)", fontweight="bold")
        ax2.set_xlim(0, 105)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "clinical_stress_tests.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved stress test figure to {out_file}")

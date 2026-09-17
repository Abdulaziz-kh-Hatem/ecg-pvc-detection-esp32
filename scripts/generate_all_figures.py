#!/usr/bin/env python3
"""
ECG PVC Detection - Figure Generation Suite.
Generates clear, high-resolution figures for visualizing the project:
- ecg_preprocessing.png: Raw vs. Bandpass Filtered (0.5-40 Hz) ECG signal
- normal_vs_pvc.png: Normal beat vs. Premature Ventricular Contraction morphology
- template_generation.png: Subject-specific median template from calibration beats
- lookahead_buffer.png: Beat segmentation and lookahead buffer diagram
- feature_importance.png: Top features selected by dual-voting (Random Forest + XGBoost)
- cohort_accuracy.png: Classification accuracy across all 15 patient records
- cohort_sensitivity.png: Classification sensitivity across all 15 patient records
- confusion_matrix.png: Confusion matrix for Patient 208 test set
- feature_selection_frequency.png: Selection frequency of candidate features across cohort
- generalization_comparison.png: Intra-patient vs. Inter-patient (unseen subject) performance
- clinical_stress_tests.png: Robustness to timing jitter and non-PVC arrhythmias
"""

import sys
import os
import json
from typing import Optional
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

sys.path.insert(0, _repo_root)

from src.data.loader import load_ecg_record
from src.signal.filtering import butter_bandpass_filter

# Clean, professional styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5
})


def plot_ecg_preprocessing(data_dir: str = "data/raw/", out_dir: str = "figures") -> None:
    """Plot raw ECG signal compared to 0.5-40 Hz bandpass filtered signal."""
    signal, _, fs = load_ecg_record("208", data_dir=data_dir, channel=0)
    start = int(200 * fs)
    raw = signal[start: start + int(4 * fs)]
    filtered = butter_bandpass_filter(raw, 0.5, 40, fs, order=3, mode="zero_phase")
    t = np.linspace(0, 4, len(raw))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t, raw, color="gray", label="Raw ECG Signal")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.set_title("ECG Preprocessing: Raw Signal", fontweight="bold")
    ax1.legend(loc="upper right")

    ax2.plot(t, filtered, color="#1f77b4", label="Filtered ECG (0.5–40 Hz Bandpass)")
    ax2.set_ylabel("Amplitude (mV)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("ECG Preprocessing: Filtered Output", fontweight="bold")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "ecg_preprocessing.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_normal_vs_pvc(data_dir: str = "data/raw/", out_dir: str = "figures") -> None:
    """Plot heartbeat morphology comparison: Normal beat vs. PVC beat."""
    signal, ann, fs = load_ecg_record("208", data_dir=data_dir, channel=0)
    filtered = butter_bandpass_filter(signal, 0.5, 40, fs, order=3, mode="zero_phase")

    # Beat 12 is N (sample 2558) and Beat 13 is V (sample 2747)
    norm_sample = 2558
    pvc_sample = 2747

    # Subplot 1: ECG context trace showing both beats
    t_start = 2200
    t_end = 3100
    ecg_chunk = filtered[t_start:t_end]
    time_chunk = (np.arange(len(ecg_chunk))) / fs

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.plot(time_chunk, ecg_chunk, color="#333333", linewidth=1.2)
    # Highlight Normal beat
    t_n_rel = (norm_sample - t_start) / fs
    ax1.plot(t_n_rel, filtered[norm_sample], marker="o", color="#2ca02c", markersize=8)
    ax1.annotate("Normal (N)", xy=(t_n_rel, filtered[norm_sample]), xytext=(t_n_rel, filtered[norm_sample] + 0.35),
                 ha="center", fontweight="bold", color="#2ca02c",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2))

    # Highlight PVC beat
    t_v_rel = (pvc_sample - t_start) / fs
    ax1.plot(t_v_rel, filtered[pvc_sample], marker="o", color="#d62728", markersize=8)
    ax1.annotate("PVC (V)", xy=(t_v_rel, filtered[pvc_sample]), xytext=(t_v_rel, filtered[pvc_sample] + 0.35),
                 ha="center", fontweight="bold", color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))

    ax1.set_title("A. Filtered ECG Rhythm Strip Showing Normal and PVC Beats (Record 208)", fontweight="bold")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude (mV)")

    # Subplot 2: Overlaid isolated beats
    pre = 36   # 100 ms
    post = 72  # 200 ms
    t_ms = (np.arange(pre + post) - pre) / fs * 1000  # ms relative to R-peak

    n_beat = filtered[norm_sample - pre: norm_sample + post]
    v_beat = filtered[pvc_sample - pre: pvc_sample + post]

    ax2.plot(t_ms, n_beat, color="#2ca02c", linewidth=2.2, label="Normal Beat (N) — Narrow QRS")
    ax2.plot(t_ms, v_beat, color="#d62728", linewidth=2.2, linestyle="--", label="PVC Beat (V) — Wide, Aberrant QRS")
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.7, label="R-Peak (t = 0 ms)")
    ax2.set_title("B. Isolated Beat Morphology Comparison (Aligned at R-Peak)", fontweight="bold")
    ax2.set_xlabel("Time Relative to R-Peak (ms)")
    ax2.set_ylabel("Amplitude (mV)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "normal_vs_pvc.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_template_generation(data_dir: str = "data/raw/", out_dir: str = "figures") -> None:
    """Plot subject-specific median template built from first 50 normal beats."""
    from src.beats.template import build_subject_template

    signal, annotation, fs = load_ecg_record("208", data_dir=data_dir, channel=0)
    filtered = butter_bandpass_filter(signal, 0.5, 40, fs, order=3, mode="causal")
    pre, post = int(0.100 * fs), int(0.200 * fs)

    median_tmpl, last_calib = build_subject_template(
        filtered, annotation.sample, np.array(annotation.symbol),
        n_beats=50, window_pre=pre, window_post=post, method="median"
    )

    beats = []
    for i in range(1, last_calib + 1):
        if annotation.symbol[i] == "N":
            idx = annotation.sample[i]
            if idx - pre >= 0 and idx + post < len(filtered):
                beats.append(filtered[idx - pre: idx + post])

    t_axis = (np.arange(len(median_tmpl)) - pre) / fs * 1000

    plt.figure(figsize=(8, 5))
    for b in beats:
        plt.plot(t_axis, b, color="gray", alpha=0.15, linewidth=1)
    plt.plot([], [], color="gray", alpha=0.3, label="Calibration Beats (First 50 Normal)")
    plt.plot(t_axis, median_tmpl, color="#D62728", linewidth=2.5, label="Subject-Specific Median Template")
    plt.axvline(0, color="gray", linestyle=":", alpha=0.7, label="R-Peak (t = 0 ms)")
    plt.title("Subject-Specific Calibration: Median Normal Beat Template (Patient 208)", fontweight="bold")
    plt.xlabel("Time Relative to R-Peak (ms)")
    plt.ylabel("Amplitude (mV)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "template_generation.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_lookahead_buffer(data_dir: str = "data/raw/", out_dir: str = "figures") -> None:
    """Plot operational logic of beat segmentation and lookahead buffer."""
    signal, annotation, fs = load_ecg_record("208", data_dir=data_dir, channel=0)
    filtered = butter_bandpass_filter(signal, 0.5, 40, fs, order=3, mode="zero_phase")
    pre, post = int(0.100 * fs), int(0.200 * fs)
    margin = int(0.30 * fs)

    symbols = annotation.symbol
    samples = annotation.sample

    pair_idx = None
    for i in range(len(symbols) - 1):
        if symbols[i] == "V" and symbols[i + 1] == "N":
            r0 = samples[i]
            r1 = samples[i + 1]
            if r0 - pre - margin >= 0 and r1 + post + margin < len(filtered):
                pair_idx = i
                break

    if pair_idx is None:
        pair_idx = 10

    r_n = samples[pair_idx]
    r_n1 = samples[pair_idx + 1]

    seg_start = max(0, r_n - pre - margin)
    seg_end = min(len(filtered), r_n1 + post + margin)
    segment = filtered[seg_start:seg_end]
    x_axis = np.arange(len(segment))
    rn_rel = r_n - seg_start
    rn1_rel = r_n1 - seg_start

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axvspan(rn_rel - pre, rn_rel + post, alpha=0.45, color="#FFD700", zorder=1, label="Current Beat (N) Window [100ms pre, 200ms post]")
    ax.axvspan(rn1_rel - pre, rn1_rel + post, alpha=0.45, color="#90EE90", zorder=2, label="Next Beat (N+1) Buffer Window")
    ax.plot(x_axis, segment, color="black", linewidth=1.3, zorder=3)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    arrow_y = y_min + y_range * 0.80
    gap_start = rn_rel + post
    gap_end = rn1_rel - pre
    mid_gap_x = (gap_start + gap_end) / 2

    ax.annotate("", xy=(gap_end, arrow_y), xytext=(gap_start, arrow_y),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.6))
    ax.text(mid_gap_x, arrow_y + y_range * 0.06, "Buffer Active\n(Awaiting Next R-Peak)",
            ha="center", va="bottom", fontsize=9, color="#222222",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#888888", alpha=0.85))

    post_rr_y = y_min + y_range * 0.10
    ax.annotate("", xy=(rn1_rel, post_rr_y), xytext=(rn_rel, post_rr_y),
                arrowprops=dict(arrowstyle="<->", color="#B22222", lw=1.4, linestyle="dashed"))
    ax.text((rn_rel + rn1_rel) / 2, post_rr_y - y_range * 0.05, "Post-RR Interval (Compensatory Pause)",
            ha="center", va="top", fontsize=8.5, color="#B22222", style="italic")

    ax.set_title("Operational Beat Segmentation & Lookahead Buffer Logic", fontweight="bold")
    ax.set_xlabel("Samples (at 360 Hz)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.set_xlim([0, len(segment) - 1])

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "lookahead_buffer.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_feature_importance(json_path: str = "results/feature_selection.json", out_dir: str = "figures") -> None:
    """Plot top features selected by dual-voting feature selection for Patient 208."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}. Run scripts/run_experiment.py first.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p208 = data["208"]
    feats = p208["selected_features"]
    rf = p208["rf_scores"]
    xgb_s = p208["xgb_scores"]

    rf_max = max(rf) if max(rf) > 0 else 1.0
    xgb_max = max(xgb_s) if max(xgb_s) > 0 else 1.0
    scores = [(rf[i]/rf_max + xgb_s[i]/xgb_max) / 2.0 * 20.0 for i in range(len(feats))]

    sorted_pairs = sorted(zip(feats, scores), key=lambda x: x[1], reverse=True)[:12]
    f_sorted = [p[0] for p in sorted_pairs][::-1]
    v_sorted = [p[1] for p in sorted_pairs][::-1]

    cmap = plt.cm.YlOrRd
    n = len(v_sorted)
    colors = [cmap(0.28 + 0.72 * (i / max(n - 1, 1))) for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(f_sorted, v_sorted, color=colors, edgecolor="black", height=0.65)
    ax.set_title("Dual-Voting Feature Importance (Top 12 Selected — Patient 208)", fontweight="bold")
    ax.set_xlabel("Consensus Importance Score (RF Gini + XGBoost Gain)")
    ax.set_ylabel("Selected Features")
    ax.set_xlim(left=0)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_cohort_performance(csv_path: str = "results/patient_results.csv", out_dir: str = "figures") -> None:
    """Plot Accuracy and Sensitivity across all 15 patient records."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}. Run scripts/run_experiment.py first.")

    df = pd.read_csv(csv_path)
    df["Patient"] = df["Patient"].astype(str)

    os.makedirs(out_dir, exist_ok=True)

    # Accuracy Plot
    plt.figure(figsize=(11, 5))
    colors_acc = ["#2ca02c" if a == 100.0 else "#1f77b4" for a in df["Accuracy"]]
    plt.bar(df["Patient"], df["Accuracy"], color=colors_acc, edgecolor="black")
    mean_acc = df["Accuracy"].mean()
    plt.axhline(y=mean_acc, color="red", linestyle="--", label=f"Cohort Mean Accuracy: {mean_acc:.2f}%")
    plt.ylim(95, 101)
    plt.title("Classification Accuracy per Patient (70/30 Chronological Split)", fontweight="bold")
    plt.xlabel("MIT-BIH Patient Record")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_acc = os.path.join(out_dir, "cohort_accuracy.png")
    plt.savefig(out_acc, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_acc}")

    # Sensitivity Plot
    plt.figure(figsize=(11, 5))
    colors_sens = ["#2ca02c" if s >= 95.0 else "#d62728" for s in df["Sensitivity"]]
    plt.bar(df["Patient"], df["Sensitivity"], color=colors_sens, edgecolor="black")
    mean_sens = df["Sensitivity"].mean()
    plt.axhline(y=mean_sens, color="blue", linestyle="--", label=f"Cohort Mean Sensitivity: {mean_sens:.2f}%")
    plt.ylim(90, 102)
    plt.title("Classification Sensitivity per Patient", fontweight="bold")
    plt.xlabel("MIT-BIH Patient Record")
    plt.ylabel("Sensitivity (%)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_sens = os.path.join(out_dir, "cohort_sensitivity.png")
    plt.savefig(out_sens, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_sens}")


def plot_confusion_matrix(cm: Optional[np.ndarray] = None, out_dir: str = "figures") -> None:
    """Plot confusion matrix for Patient 208 test set."""
    if cm is None:
        from src.beats.template import build_subject_template
        from src.data.dataset import extract_patient_dataset
        from src.features.selection import dual_voting_feature_selection
        from src.model.classifier import create_pvc_classifier, train_pvc_classifier, predict_pvc
        from sklearn.metrics import confusion_matrix

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
        X_train, y_train = X_full[:split_idx], y_full[:split_idx]
        X_test, y_test = X_full[split_idx:], y_full[split_idx:]
        top_indices, _ = dual_voting_feature_selection(X_train, y_train, n_features=12, random_state=42)
        clf = create_pvc_classifier(max_depth=8, criterion="entropy", class_weight="balanced", random_state=42)
        clf = train_pvc_classifier(clf, X_train[:, top_indices], y_train)
        y_pred = predict_pvc(clf, X_test[:, top_indices])
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (N)", "PVC (V)"])
    ax.set_yticklabels(["Normal (N)", "PVC (V)"])
    ax.set_title("Confusion Matrix — Patient 208 (Test Set)", fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > 200 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_feature_selection_frequency(json_path: str = "results/feature_selection.json", out_dir: str = "figures") -> None:
    """Plot selection frequency of candidate features across cohort."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}. Run scripts/run_experiment.py first.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_feats = []
    for pid in data:
        all_feats.extend(data[pid]["selected_features"])

    freq = Counter(all_feats)
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    feats = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    percentages = [(c / 15.0) * 100.0 for c in counts]

    colors = ["#d62728" if p >= 80.0 else "#1f77b4" for p in percentages]

    plt.figure(figsize=(13, 6))
    plt.bar(feats, percentages, color=colors, edgecolor="black")
    plt.axhline(y=80.0, color="#d62728", linestyle="--", label="High-Frequency Threshold (≥80%)")
    plt.xticks(rotation=45, ha="right", fontsize=9.5)
    plt.ylabel("Selection Frequency across Cohort (%)")
    plt.title("Feature Selection Frequency Across 15 Patients", fontweight="bold")
    plt.ylim(0, 105)
    plt.legend(loc="upper right")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "feature_selection_frequency.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_generalization_comparison(out_dir: str = "figures") -> None:
    """Plot comparison between subject-specific (intra-patient) vs. unseen subject (inter-patient LOPO)."""
    intra_csv = "results/patient_results.csv"
    inter_csv = "results/inter_patient_results.csv"

    if not os.path.exists(intra_csv) or not os.path.exists(inter_csv):
        return

    df_intra = pd.read_csv(intra_csv)
    df_inter = pd.read_csv(inter_csv)

    df_intra["Patient"] = df_intra["Patient"].astype(str)
    df_inter["Test_Patient"] = df_inter["Test_Patient"].astype(str)

    merged = pd.merge(df_intra, df_inter, left_on="Patient", right_on="Test_Patient")

    patients = merged["Patient"].tolist()
    sens_intra = merged["Sensitivity_x"].tolist()
    sens_inter = merged["Sensitivity_y"].tolist()

    x = np.arange(len(patients))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width/2, sens_intra, width, label="Subject-Specific (70/30 Split)", color="#2ca02c", edgecolor="black")
    ax.bar(x + width/2, sens_inter, width, label="Unseen Patient (Leave-One-Patient-Out)", color="#d62728", edgecolor="black")

    ax.set_ylabel("Sensitivity (%)")
    ax.set_title("Model Generalization: Subject-Specific vs. Unseen Subject Evaluation", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(patients)
    ax.legend(loc="lower left")
    ax.set_ylim(20, 105)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "generalization_comparison.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_clinical_stress_tests(json_path: str = "results/clinical_stress_test_results.json", out_dir: str = "figures") -> None:
    """Plot outcomes of jitter robustness and non-PVC arrhythmia stress tests."""
    if not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    jitter_results = data.get("rpeak_jitter_robustness_test", {})
    arrhythmia_results = data.get("arrhythmia_specificity_test", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Jitter Robustness
    if jitter_results:
        ms_vals = sorted([int(k) for k in jitter_results.keys()])
        sens_vals = [jitter_results[str(ms)]["mean_sensitivity"] for ms in ms_vals]
        spec_vals = [jitter_results[str(ms)]["mean_specificity"] for ms in ms_vals]

        ax1.plot(ms_vals, sens_vals, marker="o", color="#d62728", linewidth=2, label="Sensitivity (Recall)")
        ax1.plot(ms_vals, spec_vals, marker="s", color="#1f77b4", linewidth=2, label="Specificity")
        ax1.axvline(x=10, color="gray", linestyle=":", label="Typical Automated Detector Jitter (±10 ms)")
        ax1.set_xlabel("R-Peak Jitter Window (± ms)")
        ax1.set_ylabel("Metric (%)")
        ax1.set_title("Robustness to R-Peak Timing Jitter (Patient 208)", fontweight="bold")
        ax1.legend(loc="lower left")
        ax1.set_ylim(80, 102)

    # Plot 2: Non-PVC False Alarm Rates
    if arrhythmia_results:
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
            ax2.set_xlabel("False Alarm Rate (% Classified as PVC)")
            ax2.set_title("Misclassification on Non-PVC Arrhythmias (PACs & Fusion)", fontweight="bold")
            ax2.set_xlim(0, 105)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "clinical_stress_tests.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def main() -> None:
    print("\n" + "=" * 80)
    print(" GENERATING PROJECT FIGURES")
    print("=" * 80)
    plot_ecg_preprocessing()
    plot_normal_vs_pvc()
    plot_template_generation()
    plot_lookahead_buffer()
    plot_feature_importance()
    plot_cohort_performance()
    plot_confusion_matrix()
    plot_feature_selection_frequency()
    plot_generalization_comparison()
    plot_clinical_stress_tests()
    print("=" * 80)
    print("All figures successfully generated in figures/")


if __name__ == "__main__":
    main()

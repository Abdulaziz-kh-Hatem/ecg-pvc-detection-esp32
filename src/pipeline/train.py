"""
Canonical PVC Detection Pipeline Engine.
Executes the PVC detection pipeline (Undergraduate Research Project):
- 15 MIT-BIH Arrhythmia Database records (MLII channel)
- 3rd-order causal Butterworth bandpass filter (0.5 - 40.0 Hz)
- Subject-specific median template from first 50 normal beats
- 32 hand-crafted features per beat (108 samples: 36 pre, 72 post)
- 70/30 chronological train/test split per subject
- Dual-voting feature selection (consensus between RF Gini and XGBoost Gain, top 12)
- Balanced Decision Tree classifier (entropy criterion, max_depth=8)
- Clinical diagnostic evaluation and reporting
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, Optional
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

from src.data.loader import load_ecg_record
from src.data.dataset import extract_patient_dataset
from src.signal.filtering import butter_bandpass_filter
from src.beats.template import build_subject_template
from src.features.extraction import ALL_32_FEATURE_NAMES
from src.features.selection import dual_voting_feature_selection
from src.model.classifier import create_pvc_classifier, train_pvc_classifier, predict_pvc
from src.evaluation.metrics import compute_clinical_metrics, format_confusion_matrix

logger = logging.getLogger("canonical_pipeline")


def _setup_file_logger(log_file_path: str) -> None:
    """Ensure logging directory exists and attach a FileHandler if not already added."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file_path):
            return
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def run_canonical_pipeline(
    config_path: str = "configs/canonical_config.yaml"
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[np.ndarray], Optional[DecisionTreeClassifier]]:
    """
    Execute the verified canonical pipeline across all cohort patients.

    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file.

    Returns:
    --------
    results_df : pd.DataFrame
        Table of per-patient metrics and cohort performance.
    all_features : Dict[str, Any]
        Dictionary of selected features and scores per patient.
    p208_cm : Optional[np.ndarray]
        Confusion matrix for Patient 208.
    p208_model : Optional[DecisionTreeClassifier]
        Trained Decision Tree for Patient 208.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["dataset"]["raw_data_dir"]
    patients = [str(p) for p in cfg["dataset"]["patients"]]
    results_dir = cfg["paths"]["results_dir"]
    models_dir = cfg["paths"].get("models_dir", "models/")
    logs_dir = cfg["paths"].get("logs_dir", "logs/")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    _setup_file_logger(os.path.join(logs_dir, "canonical_pipeline.log"))
    logger.info("Starting canonical research pipeline execution.")

    results_list = []
    all_features: Dict[str, Any] = {}
    p208_cm: Optional[np.ndarray] = None
    p208_model: Optional[DecisionTreeClassifier] = None

    print("\n" + "=" * 80)
    print(" EXECUTING PVC DETECTION PIPELINE (UNDERGRADUATE RESEARCH PROJECT)")
    print("=" * 80)
    print(f"Cohort: {len(patients)} patients from MIT-BIH Arrhythmia Database")
    print("Protocol: Chronological 70/30 Split | Dual-Voting Feature Selection")
    print("-" * 80)
    print(f"{'Patient':<8} | {'Train':<6} | {'Test':<6} | {'Acc(%)':<7} | {'Sens(%)':<7} | {'Spec(%)':<7} | {'Top-3 Features'}")
    print("-" * 80)

    for pid in patients:
        signal, annotation, fs = load_ecg_record(
            record_id=pid,
            data_dir=data_dir,
            channel=cfg["dataset"]["channel"]
        )

        filtered_signal = butter_bandpass_filter(
            signal,
            lowcut=cfg["preprocessing"]["lowcut_hz"],
            highcut=cfg["preprocessing"]["highcut_hz"],
            fs=fs,
            order=cfg["preprocessing"]["filter_order"],
            mode=cfg["preprocessing"]["filter_mode"]
        )

        template, last_calib = build_subject_template(
            signal=filtered_signal,
            beats=annotation.sample,
            symbols=np.array(annotation.symbol),
            n_beats=cfg["calibration"]["calibration_beats"],
            window_pre=cfg["segmentation"]["pre_r_samples"],
            window_post=cfg["segmentation"]["post_r_samples"],
            method=cfg["calibration"]["method"]
        )

        X_full, y_full, _ = extract_patient_dataset(
            signal=filtered_signal,
            beats=annotation.sample,
            symbols=np.array(annotation.symbol),
            template=template,
            last_calib_index=last_calib,
            fs=fs,
            window_pre=cfg["segmentation"]["pre_r_samples"],
            window_post=cfg["segmentation"]["post_r_samples"]
        )

        # Chronological Split (70% Train, 30% Test)
        split_idx = int(len(X_full) * cfg["validation"]["train_ratio"])
        X_train, y_train = X_full[:split_idx], y_full[:split_idx]
        X_test, y_test = X_full[split_idx:], y_full[split_idx:]

        # Dual-Voting Feature Selection on Training Split Only
        top_indices, scores = dual_voting_feature_selection(
            X_train, y_train,
            n_features=cfg["feature_selection"]["n_selected"],
            random_state=cfg["feature_selection"]["random_forest"]["random_state"]
        )
        selected_features = [ALL_32_FEATURE_NAMES[i] for i in top_indices]

        X_train_sel = X_train[:, top_indices]
        X_test_sel = X_test[:, top_indices]

        # Train Decision Tree
        clf = create_pvc_classifier(
            max_depth=cfg["classifier"]["max_depth"],
            criterion=cfg["classifier"]["criterion"],
            class_weight=cfg["classifier"]["class_weight"],
            random_state=cfg["classifier"]["random_state"]
        )
        clf = train_pvc_classifier(clf, X_train_sel, y_train)

        # Evaluate on Test Partition
        y_pred = predict_pvc(clf, X_test_sel)
        metrics = compute_clinical_metrics(y_test, y_pred)

        acc = metrics["Accuracy"]
        sens = metrics["Sensitivity"]
        spec = metrics["Specificity"]
        cm = metrics["confusion_matrix"]

        top3_str = ", ".join(selected_features[:3])
        log_line = (f"{pid:<8} | {len(y_train):<6} | {len(y_test):<6} | {acc:<7.2f} | "
                    f"{sens:<7.2f} | {spec:<7.2f} | {top3_str}...")
        print(log_line)
        logger.info(log_line)

        patient_record = {
            "Patient": str(pid),
            "Train_N": len(y_train),
            "Test_N": len(y_test),
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": metrics["Precision"],
            "F1_Score": metrics["F1_Score"],
            "Balanced_Accuracy": metrics["Balanced_Accuracy"],
            "Tree_Depth": clf.get_depth(),
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"],
        }
        results_list.append(patient_record)

        all_features[str(pid)] = {
            "selected_features": selected_features,
            "rf_scores": scores["rf_scores"],
            "xgb_scores": scores["xgb_scores"]
        }

        if str(pid) == "208":
            p208_cm = cm
            p208_model = clf
            joblib.dump(clf, os.path.join(models_dir, "canonical_patient_208_decision_tree.joblib"))

    print("-" * 80)
    df_results = pd.DataFrame(results_list)
    cohort_summary = (
        f"Mean Accuracy:    {df_results['Accuracy'].mean():.2f}% ± {df_results['Accuracy'].std():.2f}%\n"
        f"Mean Sensitivity: {df_results['Sensitivity'].mean():.2f}% ± {df_results['Sensitivity'].std():.2f}%\n"
        f"Mean Specificity: {df_results['Specificity'].mean():.2f}% ± {df_results['Specificity'].std():.2f}%\n"
        f"Mean Precision:   {df_results['Precision'].mean():.2f}% ± {df_results['Precision'].std():.2f}%\n"
        f"Mean F1-Score:    {df_results['F1_Score'].mean():.2f}% ± {df_results['F1_Score'].std():.2f}%\n"
        f"Mean Tree Depth:  {df_results['Tree_Depth'].mean():.2f} ± {df_results['Tree_Depth'].std():.2f}"
    )
    print(cohort_summary)
    print("=" * 80)
    logger.info("Cohort Canonical Summary:\n" + cohort_summary)

    # Save canonical CSV and JSON
    csv_canonical = os.path.join(results_dir, "patient_results.csv")
    json_canonical = os.path.join(results_dir, "feature_selection.json")

    df_results.to_csv(csv_canonical, index=False)
    with open(json_canonical, "w", encoding="utf-8") as f:
        json.dump(all_features, f, indent=2)

    logger.info(f"Saved canonical outputs to {csv_canonical} and {json_canonical}")

    # Release file handler locks so files can be accessed/cleaned up safely on Windows
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    return df_results, all_features, p208_cm, p208_model

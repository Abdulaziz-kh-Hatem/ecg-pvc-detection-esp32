"""
Inter-Patient Leave-One-Patient-Out (LOPO) Evaluation Engine.
Evaluates model generalizability to completely unseen patient cohorts,
reusing the canonical signal processing, segmentation, and feature extraction components.
"""

import os
import logging
from typing import Dict, Tuple
import yaml
import numpy as np
import pandas as pd
import joblib

from src.data.loader import load_ecg_record
from src.data.dataset import extract_patient_dataset
from src.signal.filtering import butter_bandpass_filter
from src.beats.template import build_subject_template
from src.model.classifier import create_pvc_classifier, train_pvc_classifier, predict_pvc
from src.evaluation.metrics import compute_clinical_metrics
from validation.cross_validation import generate_lopo_splits

logger = logging.getLogger("lopo_pipeline")


def _setup_file_logger(log_file_path: str) -> None:
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(log_file_path):
            return
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def run_inter_patient_lopo_pipeline(
    config_path: str = "configs/canonical_config.yaml"
) -> pd.DataFrame:
    """
    Execute Leave-One-Patient-Out (LOPO) cross-validation across all cohort patients.
    Trains on 14 patients pooled and tests on the 15th unseen patient.

    Parameters:
    -----------
    config_path : str
        Path to configuration file.

    Returns:
    --------
    df_lopo : pd.DataFrame
        Table of generalization metrics per held-out patient.
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

    _setup_file_logger(os.path.join(logs_dir, "inter_patient_audit.log"))
    logger.info("Starting Inter-Patient LOPO pipeline execution.")

    print("\n" + "=" * 80)
    print(" EXECUTING INTER-PATIENT GENERALIZATION AUDIT (LEAVE-ONE-PATIENT-OUT)")
    print("=" * 80)
    print(f"Cohort: {len(patients)} patients | Protocol: 14 Patients Train -> 1 Patient Test")
    print("-" * 80)
    print(f"{'Test Patient':<12} | {'Train Beats':<11} | {'Test Beats':<10} | {'Acc(%)':<7} | {'Sens(%)':<7} | {'Spec(%)':<7} | {'Prec(%)':<7} | {'F1(%)':<7}")
    print("-" * 80)

    # Pre-extract all patient features
    patient_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
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

        X_p, y_p, _ = extract_patient_dataset(
            signal=filtered_signal,
            beats=annotation.sample,
            symbols=np.array(annotation.symbol),
            template=template,
            last_calib_index=last_calib,
            fs=fs,
            window_pre=cfg["segmentation"]["pre_r_samples"],
            window_post=cfg["segmentation"]["post_r_samples"]
        )
        patient_datasets[pid] = (X_p, y_p)

    lopo_splits = generate_lopo_splits(patients)
    lopo_results = []
    last_model = None

    for train_pids, test_pid in lopo_splits:
        X_train_list = [patient_datasets[p][0] for p in train_pids]
        y_train_list = [patient_datasets[p][1] for p in train_pids]
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        X_test, y_test = patient_datasets[test_pid]

        clf = create_pvc_classifier(
            max_depth=cfg["classifier"]["max_depth"],
            criterion=cfg["classifier"]["criterion"],
            class_weight=cfg["classifier"]["class_weight"],
            random_state=cfg["classifier"]["random_state"]
        )
        clf = train_pvc_classifier(clf, X_train, y_train)
        y_pred = predict_pvc(clf, X_test)
        last_model = clf

        metrics = compute_clinical_metrics(y_test, y_pred)
        metrics["Test_Patient"] = str(test_pid)
        metrics["Train_N"] = len(y_train)
        metrics["Test_N"] = len(y_test)
        metrics["Tree_Depth"] = clf.get_depth()
        metrics.pop("confusion_matrix")
        lopo_results.append(metrics)

        log_msg = (f"{test_pid:<12} | {len(y_train):<11} | {len(y_test):<10} | "
                   f"{metrics['Accuracy']:<7.2f} | {metrics['Sensitivity']:<7.2f} | "
                   f"{metrics['Specificity']:<7.2f} | {metrics['Precision']:<7.2f} | "
                   f"{metrics['F1_Score']:<7.2f}")
        print(log_msg)
        logger.info(log_msg)

    if last_model is not None:
        joblib.dump(last_model, os.path.join(models_dir, "lopo_global_tree.joblib"))

    df_lopo = pd.DataFrame(lopo_results)
    print("-" * 80)
    summary_str = (
        f"LOPO Mean Accuracy:    {df_lopo['Accuracy'].mean():.2f}% ± {df_lopo['Accuracy'].std():.2f}%\n"
        f"LOPO Mean Sensitivity: {df_lopo['Sensitivity'].mean():.2f}% ± {df_lopo['Sensitivity'].std():.2f}%\n"
        f"LOPO Mean Specificity: {df_lopo['Specificity'].mean():.2f}% ± {df_lopo['Specificity'].std():.2f}%\n"
        f"LOPO Mean Precision:   {df_lopo['Precision'].mean():.2f}% ± {df_lopo['Precision'].std():.2f}%\n"
        f"LOPO Mean F1-Score:    {df_lopo['F1_Score'].mean():.2f}% ± {df_lopo['F1_Score'].std():.2f}%\n"
        f"LOPO Mean Bal Acc:     {df_lopo['Balanced_Accuracy'].mean():.2f}% ± {df_lopo['Balanced_Accuracy'].std():.2f}%"
    )
    print(summary_str)
    print("=" * 80)
    logger.info("LOPO Summary:\n" + summary_str)

    out_csv = os.path.join(results_dir, "inter_patient_results.csv")
    df_lopo.to_csv(out_csv, index=False)
    logger.info(f"Saved inter-patient results to {out_csv}")

    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    return df_lopo

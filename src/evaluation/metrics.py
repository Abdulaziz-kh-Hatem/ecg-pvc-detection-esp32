"""
Evaluation metrics computation module for ECG arrhythmia classification.
Computes clinically meaningful performance indicators and confusion matrix breakdowns.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)


def compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive clinical diagnostic metrics for binary PVC classification.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground-truth binary labels (0 = Normal, 1 = PVC).
    y_pred : np.ndarray
        Predicted binary labels.

    Returns:
    --------
    metrics : Dict[str, Any]
        Dictionary of accuracy, sensitivity, specificity, precision, f1, and raw confusion matrix counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    total = tn + fp + fn + tp
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    sens = float((tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0)
    spec = float((tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0)
    prec = float((tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0)
    f1 = float(f1_score(y_true, y_pred, zero_division=0) * 100.0)
    if (tp + fn > 0) and (tn + fp > 0):
        bal_acc = float((sens + spec) / 2.0)
    elif tp + fn > 0:
        bal_acc = sens
    else:
        bal_acc = spec

    return {
        "Total_N": total,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Test_PVCs": tp + fn,
        "Test_Normals": tn + fp,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision": prec,
        "F1_Score": f1,
        "Balanced_Accuracy": bal_acc,
        "confusion_matrix": cm
    }


def format_confusion_matrix(cm: np.ndarray) -> str:
    """Format a 2x2 confusion matrix into a clean human-readable text box."""
    tn, fp, fn, tp = cm.ravel()
    return (
        f"                 Predicted Normal   Predicted PVC\n"
        f"Actual Normal:        {tn:6d}           {fp:6d}\n"
        f"Actual PVC:           {fn:6d}           {tp:6d}"
    )

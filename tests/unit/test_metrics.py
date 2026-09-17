"""
Unit tests for clinical metrics computation and formatting.
"""

import numpy as np
from src.evaluation.metrics import compute_clinical_metrics, format_confusion_matrix


def test_perfect_classification():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    m = compute_clinical_metrics(y_true, y_pred)

    assert m["Accuracy"] == 100.0
    assert m["Sensitivity"] == 100.0
    assert m["Specificity"] == 100.0
    assert m["Precision"] == 100.0
    assert m["F1_Score"] == 100.0
    assert m["Balanced_Accuracy"] == 100.0
    assert m["TN"] == 3
    assert m["TP"] == 2
    assert m["FP"] == 0
    assert m["FN"] == 0


def test_zero_positives_edge_case():
    # Dataset with only normal beats
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0])
    m = compute_clinical_metrics(y_true, y_pred)

    assert m["Accuracy"] == 100.0
    assert m["Specificity"] == 100.0
    assert m["Sensitivity"] == 0.0  # Safe zero division handling
    assert m["Precision"] == 0.0
    assert m["Test_PVCs"] == 0


def test_all_false_negatives():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])
    m = compute_clinical_metrics(y_true, y_pred)

    assert m["Accuracy"] == 0.0
    assert m["Sensitivity"] == 0.0
    assert m["FN"] == 3
    assert m["TP"] == 0


def test_confusion_matrix_formatting():
    cm = np.array([[500, 2], [5, 250]])
    formatted = format_confusion_matrix(cm)
    assert "500" in formatted
    assert "250" in formatted
